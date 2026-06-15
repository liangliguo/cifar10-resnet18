"""生成基线模型在 CIFAR-10 干净测试集上的混淆矩阵。

预处理与模型定义与 resnet-18-apgd.ipynb 保持一致（ToTensor + Normalize、
CIFAR 版 ResNet-18：conv1 改 3x3 stride1、去 maxpool），因此评估出的准确率
与 checkpoint 记录的 test_acc (0.8853) 一致。

用法：
    .venv/bin/python make_confusion_clean.py
输出：
    ppt/confusion_clean.pdf
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

CKPT = "to/dest/resnet18_cifar10_best.pth"
DATA_ROOT = "./data"
OUT = "ppt/confusion_clean.pdf"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
# 与 notebook 一致的归一化统计量
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def create_resnet18(num_classes=10):
    """CIFAR 版 ResNet-18，与 notebook 中的 create_resnet18 一致。"""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_resnet18().to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    test_ds = datasets.CIFAR10(root=DATA_ROOT, train=False,
                               download=False, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(1).cpu()
            y_true.append(y)
            y_pred.append(pred)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"clean test acc = {acc:.4f}   macro-F1 = {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    print("\nper-class accuracy:")
    for i, c in enumerate(CLASSES):
        print(f"  {c:11s} {cmn[i, i]:.3f}")

    # ---------- 绘图（行归一化混淆矩阵）----------
    plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight",
                         "font.size": 10, "axes.titlesize": 13,
                         "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Clean Test Confusion Matrix (acc={acc:.2%}, row-normalized)")
    for i in range(10):
        for j in range(10):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUT)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
