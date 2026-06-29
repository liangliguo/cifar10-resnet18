import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ---------------- Cell 0: markdown ----------------
cells.append(nbf.v4.new_markdown_cell(r"""# FGSM 对抗训练：普通模型 vs 鲁棒模型（FGSM-AT）

从 `resnet-18-apgd.ipynb` / `base_vs_robust.ipynb` 派生的**独立**子任务：把主 notebook
里的 **PGD 对抗训练**换成 **FGSM 对抗训练**（单步，FGSM-RS：带随机起点的 FGSM，
Wong et al. *Fast is better than free*, ICLR'20），得到一个鲁棒模型
`resnet18_cifar10_fgsm_at.pth`，再按 `robustness_comparison.svg` 的同款分组柱状图，
画一张「普通模型 vs FGSM-AT 鲁棒模型」在 **FGSM / PGD-20 / APGD / DeepFool** 下的
**Robust Acc** 对比图。

约定（与主 notebook 一致，勿改）：
- **攻击/训练全部在 `[0,1]` 像素空间**进行，归一化作为 `atk_model` 的第一层，避免
  `torchattacks` 的 `clamp(0,1)` 把归一化后的对抗图截毁。
- **CIFAR 版 ResNet-18 stem 改造**：`conv1`→3×3 stride1，`maxpool`→`Identity`。
- 评估口径：成对报告 **Clean Acc + Robust Acc**，攻击成功率用**条件口径**（只统计
  干净分对、被攻击翻掉的样本）；固定 ε 阶梯 FGSM < PGD-20 < APGD，robust acc 应依次下降。

> ⚠️ FGSM 单步对抗训练存在 **catastrophic overfitting**（鲁棒性突然崩到 0）的风险。
> 本 notebook 用 **随机起点 + 较大单步步长 (α=1.25·ε) + 干净/对抗混合损失 + ε 预热 +
> 梯度裁剪**的稳健配方来缓解。建议在 **GPU** 上训练（CPU 会非常慢）。

按顺序运行：① 基础环境与普通模型 → ② FGSM 对抗训练 → ③ 评估 + 画图。"""))

# ---------------- Cell 1: foundation ----------------
cells.append(nbf.v4.new_code_cell(r"""# ==================== ① 环境、普通模型加载、评估 helper（自包含）====================
# 仅保留对比所需最小集合：从主 notebook 的「优化版本」单元裁剪而来
# （去掉 t-SNE / 特征提取 / 单攻击可视化），只加载「普通(基础训练)」模型作基线。
import os
import math
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'legend.fontsize': 10,
    'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': True, 'legend.framealpha': 0.9,
})

# ---------- CIFAR 版 ResNet-18（stem 改造：3x3 stride1 + maxpool=Identity）----------
def create_resnet18(num_classes=10):
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------- 归一化包装层：放进模型，攻击在 [0,1] 空间进行 ----------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

def _find_ckpt(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"未找到权重，请把路径加入候选列表：{candidates}")

# 基础（干净训练）模型 → atk_model（普通模型 / 基线）
BASE_CKPT = _find_ckpt([
    '/kaggle/input/notebooks/liangliguo/cifar/resnet18_cifar10_best.pth',
    'to/dest/resnet18_cifar10_best.pth',
    'resnet18_cifar10_best.pth',
])
model = create_resnet18().to(device)
model.load_state_dict(torch.load(BASE_CKPT, map_location=device)['model_state_dict'])
model.eval()
atk_model = nn.Sequential(Normalize(NORM_MEAN, NORM_STD), model).to(device).eval()
print(f"✓ 普通模型已加载: {BASE_CKPT}")

# ---------- 数据根目录（保持 [0,1]，归一化交给 atk_model 内部）----------
DATA_ROOT = next((p for p in [
    '/kaggle/input/datasets/pankrzysiu/cifar10-python',   # Kaggle: pankrzysiu/cifar10-python
    './data',                                             # 本地回退
] if os.path.isdir(os.path.join(p, 'cifar-10-batches-py'))), './data')
_need_dl = not os.path.isdir(os.path.join(DATA_ROOT, 'cifar-10-batches-py'))

test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=_need_dl, transform=test_transform)
test_loader = DataLoader(Subset(test_dataset, list(range(10000))), batch_size=32, shuffle=False, num_workers=2)
print(f"✓ 测试集就绪（{len(test_loader.dataset)} 张, [0,1] 空间）")

# ---------- FGSM（[0,1] 空间，结尾 clamp，口径与 torchattacks 一致）----------
class CustomFGSM:
    def __init__(self, model, eps=8 / 255):
        self.model = model
        self.eps = eps

    def __call__(self, images, labels):
        images = images.clone().detach().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(self.model(images), labels)
        grad = torch.autograd.grad(loss, images)[0]
        return torch.clamp(images + self.eps * grad.sign(), 0, 1).detach()

# ---------- 白盒攻击评估：条件攻击成功率 + Clean/Robust Acc + L2 + ρ_adv ----------
def evaluate_attack_optimized(attack, name, fwd_model, dataloader, device, num_batches=10):
    """返回 (asr, clean_acc, robust_acc, avg_l2, avg_rho)。
      asr        = 条件攻击成功率 = #(干净分对 且 对抗分错) / #(干净分对)
      robust_acc = #(对抗分对) / 总数；avg_rho = 平均相对扰动 ‖r‖₂/‖x‖₂
    注：robust_acc 与 asr 独立，二者相加不一定 = 100%。"""
    fwd_model.eval()
    total = clean_correct = robust_correct = flipped = 0
    total_l2 = total_rho = 0.0
    eval_batches = min(num_batches, len(dataloader))
    for i, (imgs, lbls) in enumerate(tqdm(dataloader, total=eval_batches,
                                          desc=f"Evaluating {name} (≈{eval_batches * dataloader.batch_size} 张)")):
        if i >= num_batches:
            break
        imgs, lbls = imgs.to(device), lbls.to(device)
        adv = attack(imgs, lbls).detach()
        with torch.no_grad():
            clean_ok = (fwd_model(imgs).argmax(1) == lbls)
            adv_ok = (fwd_model(adv).argmax(1) == lbls)
            clean_correct += clean_ok.sum().item()
            robust_correct += adv_ok.sum().item()
            flipped += (clean_ok & ~adv_ok).sum().item()
            total += lbls.size(0)
            r = torch.norm((adv - imgs).view(imgs.size(0), -1), dim=1)
            total_l2 += r.sum().item()
            total_rho += (r / torch.norm(imgs.view(imgs.size(0), -1), dim=1).clamp_min(1e-12)).sum().item()
        del imgs, lbls, adv
    return (flipped / max(1, clean_correct), clean_correct / total,
            robust_correct / total, total_l2 / total, total_rho / total)

print("\n✓ helper 就绪：CustomFGSM / evaluate_attack_optimized")"""))

# ---------------- Cell 2: FGSM adversarial training ----------------
cells.append(nbf.v4.new_code_cell(r"""# ==================== ② FGSM 对抗训练（FGSM-RS：随机起点的单步 FGSM）====================
# 说明：
#   - 防御思路与 PGD-AT 相同（用现场生成的对抗样本微调模型），但内层攻击由 PGD-7 换成
#     【单步 FGSM】，训练快得多。为缓解 catastrophic overfitting，采用稳健配方：
#       · 随机起点 δ~U(-ε,ε)          （FGSM-RS 的关键，纯 FGSM-AT 易崩）
#       · 单步步长 α = 1.25·ε         （Wong et al. 推荐）
#       · 干净/对抗混合损失 (1-λ)·CE(clean)+λ·CE(adv)   兜底防塌缩
#       · ε 预热 + 梯度裁剪
#   - 基于已加载的干净 model 克隆一份做对抗微调，原 model / atk_model 保持不变作基线。
#   - 若已有 resnet18_cifar10_fgsm_at.pth，则默认直接加载、跳过训练。
#   - ⚠️ 运行前请先运行 ① 单元。建议 GPU。
ROBUST_CKPT = "resnet18_cifar10_fgsm_at.pth"   # FGSM-AT 鲁棒模型保存/加载路径

# ---------- 超参数 ----------
TRAIN_SUBSET  = None       # None = 全量训练集(50000)；也可填整数只取子集（快速试跑）
EPOCHS        = 30         # 单步 AT 较省，30 轮足够看出鲁棒性
BATCH_SIZE    = 128
EPS           = 8 / 255    # Linf 扰动预算（与评估一致）
ALPHA         = 1.25 * EPS # FGSM-RS 单步步长（> ε，配合随机起点）
WARMUP_EPOCHS = 5          # ε 预热轮数
LR            = 0.1        # SGD + 多步衰减
MILESTONES    = [15, 25]   # 在这些 epoch 把 LR ÷10
LR_GAMMA      = 0.1
LAMBDA_ADV    = 0.5        # 损失 = (1-λ)·CE(clean) + λ·CE(adv)
CLIP_NORM     = 1.0
SEED          = 42
SKIP_AT_TRAINING = True    # True 且 ckpt 存在 → 直接加载；否则从头训练

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------- 克隆普通模型做鲁棒微调（原 model / atk_model 保持基线）----------
robust_net = copy.deepcopy(model).to(device)
atk_model_robust = nn.Sequential(Normalize(NORM_MEAN, NORM_STD), robust_net).to(device)

# ---------- FGSM-RS 攻击（Linf, [0,1] 空间，随机起点 + 单步）----------
def fgsm_rs_attack(fwd_model, imgs, labels, eps, alpha):
    was_training = fwd_model.training
    fwd_model.eval()
    delta = torch.empty_like(imgs).uniform_(-eps, eps)
    x_adv = torch.clamp(imgs + delta, 0, 1).detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(fwd_model(x_adv), labels)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + alpha * grad.sign()
    x_adv = torch.min(torch.max(x_adv, imgs - eps), imgs + eps)  # 投影回 Linf 球
    x_adv = torch.clamp(x_adv, 0, 1).detach()                    # 裁剪到合法图像
    if was_training:
        fwd_model.train()
    return x_adv

history = {"epoch": [], "loss": [], "clean_acc": [], "adv_acc": [], "lr": []}

if SKIP_AT_TRAINING and os.path.exists(ROBUST_CKPT):
    # ===== 跳过训练：直接加载已有 FGSM-AT 鲁棒模型 =====
    print(f"⏭️  跳过 FGSM 对抗训练，直接加载: {ROBUST_CKPT}")
    robust_net.load_state_dict(torch.load(ROBUST_CKPT, map_location=device)['model_state_dict'])
    robust_net.eval()
    print("✓ FGSM-AT 鲁棒模型加载完成")
else:
    # ===== 从头做 FGSM 对抗训练 =====
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # → [0,1]，归一化交给 atk_model_robust 内部
    ])
    train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True,
                                     download=_need_dl, transform=train_transform)
    if TRAIN_SUBSET is None:
        train_data = train_dataset
    else:
        _g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(len(train_dataset), generator=_g)[:TRAIN_SUBSET].tolist()
        train_data = Subset(train_dataset, idx)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    print(f"ℹ 单卡训练，device={device}，batch={BATCH_SIZE}，样本数={len(train_data)}")

    optimizer = torch.optim.SGD(robust_net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=LR_GAMMA)
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print(f"FGSM 对抗训练(FGSM-RS)：epochs={EPOCHS}, eps={EPS:.4f}, alpha={ALPHA:.4f}, "
          f"LR={LR}, milestones={MILESTONES}, lambda_adv={LAMBDA_ADV}")
    for epoch in range(1, EPOCHS + 1):
        # ε 预热：前 WARMUP_EPOCHS 轮从 0.5×eps 线性升到 1×eps，步长随 eps 缩放
        cur_eps = EPS * (0.5 + 0.5 * min(1.0, (epoch - 1) / max(1, WARMUP_EPOCHS - 1)))
        cur_alpha = 1.25 * cur_eps

        robust_net.train()
        run_loss, clean_correct, adv_correct, seen = 0.0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (eps={cur_eps:.4f})")
        for imgs, lbls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            # 1) 用当前模型现场把这批干净图打成单步对抗样本
            x_adv = fgsm_rs_attack(atk_model_robust, imgs, lbls, eps=cur_eps, alpha=cur_alpha)
            # 2) 干净 + 对抗 混合损失（标签都是原始 y）
            robust_net.train()
            optimizer.zero_grad()
            out_clean = atk_model_robust(imgs)
            out_adv = atk_model_robust(x_adv)
            loss = (1 - LAMBDA_ADV) * criterion(out_clean, lbls) + LAMBDA_ADV * criterion(out_adv, lbls)
            loss.backward()
            clip_grad_norm_(robust_net.parameters(), CLIP_NORM)
            optimizer.step()
            run_loss += loss.item() * lbls.size(0)
            clean_correct += (out_clean.argmax(1) == lbls).sum().item()
            adv_correct += (out_adv.argmax(1) == lbls).sum().item()
            seen += lbls.size(0)
            pbar.set_postfix(loss=f"{run_loss/seen:.3f}",
                             clean=f"{clean_correct/seen:.2%}",
                             adv=f"{adv_correct/seen:.2%}")
            del imgs, lbls, x_adv, out_clean, out_adv, loss
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: loss={run_loss/seen:.4f}, "
              f"clean-acc={clean_correct/seen:.2%}, adv-acc={adv_correct/seen:.2%}, lr={cur_lr:.5f}")
        history["epoch"].append(epoch); history["loss"].append(run_loss / seen)
        history["clean_acc"].append(clean_correct / seen); history["adv_acc"].append(adv_correct / seen)
        history["lr"].append(cur_lr)
        # catastrophic overfitting 预警：训练 adv-acc 突然暴涨（鲁棒性其实崩了）
        if epoch >= 2 and history["adv_acc"][-1] - history["adv_acc"][-2] > 0.20:
            print("  ⚠️ adv-acc 单轮暴涨 >20%，可能发生 catastrophic overfitting，"
                  "建议减小 alpha 或换 cyclic-LR。")
        scheduler.step()

    robust_net.eval()
    torch.save({'model_state_dict': robust_net.state_dict()}, ROBUST_CKPT)
    print(f"\n✓ FGSM-AT 鲁棒模型已保存：{ROBUST_CKPT}")

    # ---------- 训练曲线（clean-acc / adv-acc / loss）→ 矢量图 ----------
    ep = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    l1, = ax1.plot(ep, [a * 100 for a in history["clean_acc"]], color="#4C72B0",
                   marker="o", ms=3, lw=1.8, label="Train clean-acc")
    l2, = ax1.plot(ep, [a * 100 for a in history["adv_acc"]], color="#DD8452",
                   marker="s", ms=3, lw=1.8, label="Train adv-acc (FGSM)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    l3, = ax2.plot(ep, history["loss"], color="#8C8C8C", lw=1.4, ls="--", label="Train loss")
    ax2.set_ylabel("Loss"); ax2.grid(False)
    for ms in MILESTONES:
        if ms <= EPOCHS:
            ax1.axvline(ms, color="0.6", ls=":", lw=1)
    ax1.set_title("FGSM-AT Training Curves: clean vs adversarial accuracy", pad=12)
    ax1.legend(handles=[l1, l2, l3], loc="lower right")
    fig.tight_layout()
    fig.savefig("fgsm_at_training_curves.svg"); fig.savefig("fgsm_at_training_curves.pdf")
    print("✓ 训练曲线矢量图已保存：fgsm_at_training_curves.svg / .pdf")
    plt.show()"""))

# ---------------- Cell 3: evaluation + chart ----------------
cells.append(nbf.v4.new_code_cell(r"""# ==================== ③ 普通 vs FGSM-AT 鲁棒：Robust Acc 对比 + 画图 ====================
# 说明：
#   - 攻击由弱到强：FGSM(单步) < PGD-20(标准基准) < APGD(自适应)；DeepFool 求最小扰动、
#     无 ε 预算，迭代到刚好越界，几乎必然成功，故两模型 Robust Acc 都≈0（作对照档）。
#   - 每种攻击都按「白盒」对被评估模型单独构造；统一 Clean/Robust Acc + 条件攻击成功率口径。
#   - 把四种攻击下两模型的 Robust Acc 画成同一张分组柱状图（与 robustness_comparison.svg
#     同款），存 fgsm_at_robustness_comparison.svg / .pdf。
#   - ⚠️ 运行前请先运行 ① 和 ② 单元。
import numpy as np
import matplotlib.pyplot as plt
import torchattacks

def _build_fgsm(fm):
    return CustomFGSM(fm, eps=8 / 255)                                  # 单步弱攻击

def _build_pgd20(fm):
    return torchattacks.PGD(fm, eps=8/255, alpha=2/255, steps=20, random_start=True)

def _build_apgd(fm):
    return torchattacks.APGD(fm, norm='Linf', eps=8/255, steps=20, n_restarts=1, loss='ce')

def _build_deepfool(fm):
    return torchattacks.DeepFool(fm, steps=50, overshoot=0.02)          # 最小 L2 扰动, 无 ε 预算

EVAL_BATCHES = len(test_loader)   # 固定 ε 攻击：全量 10000
DF_BATCHES   = 32                 # DeepFool 慢：约 1000 张

# (显示名, builder, 短名, 评估 batch 数)
ATTACKS = [("FGSM   (单步, 弱)",        _build_fgsm,     "FGSM",     EVAL_BATCHES),
           ("PGD-20 (Linf 8/255)",     _build_pgd20,    "PGD-20",   EVAL_BATCHES),
           ("APGD   (Linf 8/255, 强)", _build_apgd,     "APGD",     EVAL_BATCHES),
           ("DeepFool (最小扰动, 无ε)", _build_deepfool, "DeepFool", DF_BATCHES)]
MODELS = [("普通模型", atk_model), ("鲁棒模型", atk_model_robust)]

# results[short][模型tag] = (clean_acc, robust_acc, asr)
results = {short: {} for _, _, short, _ in ATTACKS}

print("=" * 66)
print("普通模型 vs FGSM-AT 鲁棒模型：Clean / Robust Acc 对比")
for atk_name, builder, short, n_batches in ATTACKS:
    print("-" * 66)
    print(f"[{atk_name}]")
    print(f"{'模型':<8}{'Clean Acc':>12}{'Robust Acc':>12}{'攻击成功率':>12}")
    for tag, fm in MODELS:
        atk = builder(fm)
        asr, clean_acc, rob_acc, _, _ = evaluate_attack_optimized(
            atk, atk_name, fm, test_loader, device, num_batches=n_batches)
        print(f"{tag:<8}{clean_acc:>11.2%}{rob_acc:>12.2%}{asr:>12.2%}")
        results[short][tag] = (clean_acc, rob_acc, asr)
print("-" * 66)
print("解读：")
print("- 对抗训练若有效：FGSM-AT 鲁棒模型在各攻击下 Robust Acc 都应高于普通模型。")
print("- 固定 ε 阶梯 FGSM < PGD-20 < APGD：同一模型 Robust Acc 应依次下降；")
print("  若 FGSM 上很高但 PGD-20/APGD 上骤降，是 FGSM-AT 典型的『假鲁棒/梯度混淆』征兆。")
print("- DeepFool 无 ε 预算、迭代到刚好越界 → 几乎必然成功，故两模型 Robust Acc 都≈0。")
print("=" * 66)

# ---------- 分组柱状图（与 robustness_comparison.svg 同款）----------
attack_order = [short for _, _, short, _ in ATTACKS]
x = np.arange(len(attack_order))
width = 0.36
plain_rob = [results[s]["普通模型"][1] * 100 for s in attack_order]
robust_rob = [results[s]["鲁棒模型"][1] * 100 for s in attack_order]
plain_clean = results[attack_order[0]]["普通模型"][0] * 100   # clean 与攻击无关，取其一
robust_clean = results[attack_order[0]]["鲁棒模型"][0] * 100

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width / 2, plain_rob, width, label="Standard model", color="#8C8C8C", edgecolor="white")
b2 = ax.bar(x + width / 2, robust_rob, width, label="Robust model (FGSM-AT)", color="#C44E52", edgecolor="white")
ax.axhline(plain_clean, color="#8C8C8C", ls="--", lw=1, alpha=0.8)
ax.axhline(robust_clean, color="#C44E52", ls="--", lw=1, alpha=0.8)
ax.text(len(attack_order) - 0.5, plain_clean + 1, f"Standard clean {plain_clean:.0f}%",
        color="#5A5A5A", fontsize=8, ha="right")
ax.text(len(attack_order) - 0.5, robust_clean + 1, f"Robust clean {robust_clean:.0f}%",
        color="#C44E52", fontsize=8, ha="right")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(attack_order)
ax.set_ylabel("Robust Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Robust Accuracy: Standard vs FGSM-AT (FGSM / PGD-20 / APGD / DeepFool)", pad=12)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("fgsm_at_robustness_comparison.svg")
fig.savefig("fgsm_at_robustness_comparison.pdf")
print("✓ 鲁棒性对比矢量图已保存：fgsm_at_robustness_comparison.svg / .pdf")
plt.show()"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
with open('fgsm_at.ipynb', 'w') as f:
    nbf.write(nb, f)
print("wrote fgsm_at.ipynb with", len(cells), "cells")
