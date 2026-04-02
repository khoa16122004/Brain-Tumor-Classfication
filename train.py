import os
import random
from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torchvision.utils import save_image

from utils import *
from tqdm import tqdm
from config import *
from dataset import BrainTumorDataset
from architech import *
import logging

# Uncomment and use argparse if needed
# import argparse
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                     help='initial learning rate', dest='lr')
# parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
# parser.add_argument('--epochs', default=10, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--arch', type=str, choices=CLASSIFIERS_ARCHITECTURES)
# parser.add_argument('--dataset', type=str, choices=DATASETS)
# parser.add_argument('--optimizer', default='Adam', type=str,
#                     help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
# parser.add_argument('--gpu', default=None, type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')

# args = parser.parse_args()

# Use the config or argparse arguments to set these variables
# lr = args.lr
# epochs = args.epochs
# device = args.gpu if args.gpu else 'cuda:0'
# OUTDIR_TRAIN = args.outdir

os.environ["CUDA_VISIBLE_DEVICES"] = device


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_gradient_map(model, input_tensor, target_class=None):
    model.eval()

    x = input_tensor.clone().detach()
    x.requires_grad_(True)
    model.zero_grad()

    output = model(x)
    output_logits = output.detach()

    if target_class is None:
        target_class = output.argmax(dim=1)

    score = output.gather(1, target_class.view(-1, 1)).sum()
    score.backward()

    grad = x.grad
    saliency = grad.abs().sum(dim=1)

    h, w = saliency.shape[-2:]
    saliency = (h * w) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1, 1, 1) + 1e-8)

    return saliency.detach(), output_logits


def integrated_gradients(model, input_tensor, target_class=None, steps=32, baseline=None):
    model.eval()

    x = input_tensor.clone().detach()
    batch_size = x.size(0)

    if baseline is None:
        baseline = torch.zeros_like(x)

    if target_class is None:
        with torch.no_grad():
            target_class = model(x).argmax(dim=1)

    grads = torch.zeros_like(x)
    output_logits = None

    for i in range(1, steps + 1):
        alpha = float(i) / steps
        inp = baseline + alpha * (x - baseline)
        inp.requires_grad_(True)

        model.zero_grad()
        output = model(inp)
        output_logits = output.detach()

        score = output.gather(1, target_class.view(-1, 1)).sum()
        score.backward()
        grads += inp.grad.detach()

    avg_grad = grads / steps
    ig = (x - baseline) * avg_grad

    saliency = ig.abs().sum(dim=1)
    h, w = saliency.shape[-2:]
    saliency = (h * w) * saliency / (saliency.view(batch_size, -1).sum(dim=1).view(-1, 1, 1) + 1e-8)

    return saliency.detach(), output_logits


def normalize_per_sample(tensor):
    min_val = tensor.amin(dim=(-2, -1), keepdim=True)
    max_val = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def heat_to_color(heat):
    # Quick RGB mapping without extra dependencies.
    red = heat
    green = torch.clamp(1.0 - (heat - 0.5).abs() * 2.0, min=0.0, max=1.0)
    blue = 1.0 - heat
    return torch.stack([red, green, blue], dim=0)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device_torch):
    model.eval()

    losses_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device_torch)
        labels = labels.to(device_torch)

        output = model(imgs)
        loss = criterion(output, labels)
        acc = accuracy(output, labels)

        losses_meter.update(loss.item(), imgs.shape[0])
        acc_meter.update(acc[0].item(), imgs.shape[0])

        all_preds.append(output.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    return {
        "loss": losses_meter.avg,
        "acc": acc_meter.avg,
        "preds": preds,
        "labels": labels,
    }


def compute_metrics(labels, preds, num_classes):
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for gt, pd in zip(labels, preds):
        conf_mat[gt.long(), pd.long()] += 1

    class_metrics = []
    eps = 1e-8
    for cls_idx in range(num_classes):
        tp = conf_mat[cls_idx, cls_idx].item()
        fp = conf_mat[:, cls_idx].sum().item() - tp
        fn = conf_mat[cls_idx, :].sum().item() - tp
        support = conf_mat[cls_idx, :].sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)

        class_metrics.append(
            {
                "class": cls_idx,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro_precision = sum(c["precision"] for c in class_metrics) / num_classes
    macro_recall = sum(c["recall"] for c in class_metrics) / num_classes
    macro_f1 = sum(c["f1"] for c in class_metrics) / num_classes

    overall_acc = (conf_mat.trace().item() / (conf_mat.sum().item() + eps)) * 100.0

    return {
        "confusion_matrix": conf_mat,
        "class_metrics": class_metrics,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "overall_acc": overall_acc,
    }


def save_saliency_examples(model, loader, device_torch, outdir, max_samples=8):
    model.eval()
    outdir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for imgs, labels in loader:
        imgs = imgs.to(device_torch)
        labels = labels.to(device_torch)

        with torch.no_grad():
            preds = model(imgs).argmax(dim=1)

        grad_saliency, _ = simple_gradient_map(model, imgs, target_class=preds)
        ig_saliency, _ = integrated_gradients(model, imgs, target_class=preds, steps=32)

        grad_saliency = normalize_per_sample(grad_saliency)
        ig_saliency = normalize_per_sample(ig_saliency)

        for i in range(imgs.size(0)):
            if saved >= max_samples:
                return

            img = imgs[i].detach().cpu()
            if img.dim() == 2:
                img = img.unsqueeze(0)
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)

            grad_heat = grad_saliency[i].detach().cpu()
            ig_heat = ig_saliency[i].detach().cpu()

            grad_color = heat_to_color(grad_heat)
            ig_color = heat_to_color(ig_heat)

            grad_overlay = torch.clamp(0.6 * img + 0.4 * grad_color, 0.0, 1.0)
            ig_overlay = torch.clamp(0.6 * img + 0.4 * ig_color, 0.0, 1.0)

            pred_label = int(preds[i].item())
            gt_label = int(labels[i].item())

            save_image(img, outdir / f"sample_{saved:02d}_raw_gt{gt_label}_pred{pred_label}.png")
            save_image(grad_overlay, outdir / f"sample_{saved:02d}_grad_overlay_gt{gt_label}_pred{pred_label}.png")
            save_image(ig_overlay, outdir / f"sample_{saved:02d}_ig_overlay_gt{gt_label}_pred{pred_label}.png")

            saved += 1


def run_test_and_report(model, test_loader, criterion, device_torch, outdir, saliency_samples=8):
    final_test = evaluate_model(model, test_loader, criterion, device_torch)
    num_classes = int(final_test["preds"].max().item()) + 1
    metrics = compute_metrics(final_test["labels"], final_test["preds"], num_classes=num_classes)

    lines = []
    lines.append("===== Test Performance =====")
    lines.append(f"Loss: {final_test['loss']:.4f}")
    lines.append(f"Accuracy: {metrics['overall_acc']:.2f}%")
    lines.append(f"Macro Precision: {metrics['macro_precision']:.4f}")
    lines.append(f"Macro Recall: {metrics['macro_recall']:.4f}")
    lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")
    lines.append("\nPer-class metrics:")
    for m in metrics["class_metrics"]:
        lines.append(
            f"Class {m['class']}: Precision={m['precision']:.4f}, "
            f"Recall={m['recall']:.4f}, F1={m['f1']:.4f}, Support={m['support']}"
        )

    lines.append("\nConfusion Matrix (rows=GT, cols=Pred):")
    conf_mat = metrics["confusion_matrix"]
    for row in conf_mat.tolist():
        lines.append(" ".join(str(v) for v in row))

    report = "\n".join(lines)
    print(report)
    logging.info("\n" + report)

    with open(outdir / "test_performance.txt", "w") as f:
        f.write(report)

    saliency_dir = outdir / "saliency_test"
    save_saliency_examples(model, test_loader, device_torch, saliency_dir, max_samples=saliency_samples)
    print(f"Saved saliency examples to: {saliency_dir}")
    logging.info(f"Saved saliency examples to: {saliency_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train/Test brain tumor classifier")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run evaluation on test set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for test-only mode")
    parser.add_argument("--saliency-samples", type=int, default=8, help="Number of test samples to visualize")
    args = parser.parse_args()

    set_seed(42)

    outdir = Path(OUTDIR_TRAIN)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(outdir / "training_log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training process")

    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_torch}")

    train_dataset = BrainTumorDataset(ARGUMENT_PATH, ARGUMENT_DIR)
    test_dataset = BrainTumorDataset(TESTING_ANNOTATION_PATH, TESTING_NEW_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet50().to(device_torch)
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device_torch)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if args.test_only:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else (outdir / "best.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device_torch))
        logging.info(f"Loaded checkpoint for test-only mode: {ckpt_path}")
        run_test_and_report(model, test_loader, criterion, device_torch, outdir, saliency_samples=args.saliency_samples)
        return

    best_test_acc = -1.0

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        losses_meter = AverageMeter()
        acc_meter = AverageMeter()

        for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
            imgs = imgs.to(device_torch)
            labels = labels.to(device_torch)

            output = model(imgs)
            loss = criterion(output, labels)
            acc = accuracy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_meter.update(loss.item(), imgs.shape[0])
            acc_meter.update(acc[0].item(), imgs.shape[0])

        scheduler.step()

        test_stats = evaluate_model(model, test_loader, criterion, device_torch)
        epoch_log = (
            f"Epoch {epoch:03d} | "
            f"Train Loss: {losses_meter.avg:.4f} | Train Acc: {acc_meter.avg:.2f}% | "
            f"Test Loss: {test_stats['loss']:.4f} | Test Acc: {test_stats['acc']:.2f}%"
        )
        print(epoch_log)
        logging.info(epoch_log)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), outdir / f"ep{epoch}.pth")

        if test_stats["acc"] > best_test_acc:
            best_test_acc = test_stats["acc"]
            torch.save(model.state_dict(), outdir / "best.pth")

    best_ckpt = outdir / "best.pth"
    model.load_state_dict(torch.load(best_ckpt, map_location=device_torch))
    run_test_and_report(model, test_loader, criterion, device_torch, outdir, saliency_samples=args.saliency_samples)


if __name__ == "__main__":
    main()