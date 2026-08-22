"""Fine-tuning do GoogLeNet num conjunto próprio, para dreams autorais.

Treinar o classificador num domínio novo reescreve as características das
camadas intermediárias — e são elas que o DeepDream amplifica. Treinado em
ilustração botânica, o modelo passa a fazer brotar pétalas e folhas em vez de
focinhos de cachorro.

    .venv/bin/python dataset.py --category "Botanical illustrations by family" ...
    .venv/bin/python train.py dataset/botanical -o modelos/botanico.pt

O checkpoint resultante é aceito onde quer que um modelo seja pedido:

    .venv/bin/python deepdream.py foto.jpg -m modelos/botanico.pt

O pré-processamento é **o mesmo** do `dream()` — Caffe, BGR, médias 104/117/123.
Treinar com a normalização da ImageNet e sonhar com a do Caffe produziria
características inconsistentes com o que o motor espera.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from deepdream import CAFFE_ORDER, load_model, pick_device

# Até onde congelar. As primeiras camadas aprendem bordas e cor, que não mudam
# entre domínios; deixá-las fixas acelera e estabiliza. O DeepDream trabalha de
# `inception_4c` para frente, então é dali que interessa treinar.
DEFAULT_FREEZE_UNTIL = "inception_3b"


def build_loaders(root, caffe_transform, batch_size, val_split, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        caffe_transform,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        caffe_transform,
    ])

    full = ImageFolder(root)
    val_size = max(1, int(len(full) * val_split))
    train_set, val_set = random_split(
        full, [len(full) - val_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    # random_split devolve views do mesmo dataset, então cada metade precisa da
    # sua própria transformação — senão a validação também sofre augmentação.
    train_set.dataset = ImageFolder(root, transform=train_tf)
    val_set.dataset = ImageFolder(root, transform=val_tf)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(val_set, batch_size=batch_size, num_workers=workers),
        full.classes,
    )


def freeze_through(model, until):
    """Congela tudo até `until`, inclusive. Devolve o que ficou treinável."""
    if until not in CAFFE_ORDER:
        raise ValueError(f"Bloco desconhecido: {until}")
    cut = CAFFE_ORDER.index(until)

    trainable = []
    for index, name in enumerate(CAFFE_ORDER):
        block = getattr(model, name, None)
        if block is None or not hasattr(block, "parameters"):
            continue
        for p in block.parameters():
            p.requires_grad_(index > cut)
        if index > cut:
            trainable.append(name)
    for p in model.loss3_classifier.parameters():
        p.requires_grad_(True)
    return trainable + ["loss3_classifier"]


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total)


def train(root, output, epochs, batch_size, lr, val_split, freeze_until,
          workers, device=None):
    device = pick_device(device)
    base, caffe_transform = load_model("bvlc", device)

    train_loader, val_loader, classes = build_loaders(
        root, caffe_transform, batch_size, val_split, workers
    )
    print(f"{len(classes)} classes, {len(train_loader.dataset)} treino / "
          f"{len(val_loader.dataset)} validação, em {device.type}")

    # Cabeça nova: o conjunto tem outro número de classes que a ImageNet.
    base.loss3_classifier = nn.Linear(1024, len(classes)).to(device)
    trainable = freeze_through(base, freeze_until)
    print(f"treinando: {', '.join(trainable)}")

    params = [p for p in base.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best = 0.0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        base.train()
        started, running, seen = time.time(), 0.0, 0

        for index, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(base(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running += loss.item() * labels.numel()
            seen += labels.numel()
            print(f"\répoca {epoch}/{epochs}  lote {index}/{len(train_loader)}  "
                  f"perda {running / seen:.3f}", end="", flush=True)

        schedule.step()
        accuracy = evaluate(base, val_loader, device)
        print(f"  —  acerto {accuracy:.1%}  ({time.time() - started:.0f}s)")

        if accuracy >= best:
            best = accuracy
            torch.save({"arch": "bvlc", "classes": classes,
                        "accuracy": accuracy,
                        "state_dict": base.state_dict()}, output)

    print(f"\nMelhor acerto: {best:.1%}")
    print(f"Checkpoint em {output}")
    print(f"Use com: .venv/bin/python deepdream.py foto.jpg -m {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning do GoogLeNet.")
    parser.add_argument("dataset", help="Pasta com uma subpasta por classe")
    parser.add_argument("-o", "--output", default="modelos/custom.pt")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--freeze-until", default=DEFAULT_FREEZE_UNTIL,
                        help="Congela tudo até este bloco, inclusive")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    train(args.dataset, args.output, args.epochs, args.batch_size, args.lr,
          args.val_split, args.freeze_until, args.workers, args.device)


if __name__ == "__main__":
    main()
