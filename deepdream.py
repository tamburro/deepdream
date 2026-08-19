"""DeepDream clássico (Google/Mordvintsev, 2015) em PyTorch, acelerado por MPS.

Reproduz o notebook original em Caffe o mais fielmente possível: pesos
bvlc_googlenet originais, pré-processamento Caffe (BGR, média subtraída,
escala 0-255), objetivo L2, jitter por passo, passo normalizado pela média
absoluta do gradiente e pirâmide de octaves com reinjeção de detalhe.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_caffe_models import (
    googlenet_bvlc,
    googlenet_places205,
    googlenet_places365,
)
from torchvision.models import GoogLeNet_Weights
from torchvision.models import googlenet as torchvision_googlenet

# Ordem exata em que o GoogLeNet do Caffe aplica seus módulos.
CAFFE_ORDER = [
    "conv1", "relu1", "pool1", "norm1",
    "conv2_reduce", "relu2_reduce", "conv2", "relu2", "norm2", "pool2",
    "inception_3a", "inception_3b", "pool3",
    "inception_4a", "inception_4b", "inception_4c", "inception_4d", "inception_4e", "pool4",
    "inception_5a", "inception_5b",
]

# O GoogLeNet do torchvision usa batchnorm no lugar do LRN e nomeia sem underscore.
TORCHVISION_ORDER = [
    "conv1", "maxpool1", "conv2", "conv3", "maxpool2",
    "inception3a", "inception3b", "maxpool3",
    "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "maxpool4",
    "inception5a", "inception5b",
]

# Nomes canônicos de camada, no formato Caffe, válidos em todos os backends.
INCEPTION_BLOCKS = [b for b in CAFFE_ORDER if b.startswith("inception")]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TorchvisionPreprocess(nn.Module):
    """Normalização ImageNet + o transform_input embutido no googlenet do torchvision."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return torch.cat(
            [
                x[0:1] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5,
                x[1:2] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5,
                x[2:3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5,
            ],
            dim=0,
        )


def _load_torchvision():
    model = torchvision_googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    return model, TorchvisionPreprocess()


MODELS = {
    # ImageNet, pesos Caffe originais — é este que reproduz o DeepDream de 2015.
    "bvlc": (googlenet_bvlc, CAFFE_ORDER, lambda name: name),
    # Cenários: templos, arcos, arquitetura.
    "places365": (googlenet_places365, CAFFE_ORDER, lambda name: name),
    "places205": (googlenet_places205, CAFFE_ORDER, lambda name: name),
    # Outro treinamento, com batchnorm: resultado mais suave e menos "puppy slug".
    "torchvision": (_load_torchvision, TORCHVISION_ORDER, lambda name: name.replace("_", "")),
}

# Padrões do notebook original.
DEFAULT_MODEL = "bvlc"
DEFAULT_LAYER = "inception_4c/output"
DEFAULT_ITERATIONS = 10
DEFAULT_STEP_SIZE = 1.5  # em unidades de pixel 0-255, como no Caffe
DEFAULT_OCTAVES = 4
DEFAULT_OCTAVE_SCALE = 1.4
DEFAULT_JITTER = 32

_CACHE = {}


def pick_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_layer(name):
    """Aceita 'inception_4c/output', 'inception_4c', 'inception4c' ou '4c'."""
    key = name.strip().lower().split("/")[0]
    if not key.startswith("inception"):
        key = f"inception_{key}"
    key = key.replace("inception", "inception_").replace("__", "_")
    if key in INCEPTION_BLOCKS:
        return key
    raise ValueError(
        f"Camada desconhecida: {name!r}. Disponíveis: "
        + ", ".join(f"{b}/output" for b in INCEPTION_BLOCKS)
    )


class FeatureExtractor(nn.Module):
    """Roda o GoogLeNet até o bloco mais profundo pedido e devolve as ativações."""

    def __init__(self, model_name, layers, device):
        super().__init__()
        loader, order, to_attr = MODELS[model_name]
        targets = [to_attr(resolve_layer(name)) for name in layers]
        indices = sorted({order.index(t) for t in targets})

        if model_name not in _CACHE:
            _CACHE[model_name] = loader()
        base, transform = _CACHE[model_name]
        base.eval()
        # O transform do torchvision tem buffers, então também precisa ser cópia.
        self.transform = copy.deepcopy(transform)

        # Cópia: o cache guarda o modelo em CPU e os blocos movidos para a GPU
        # precisam ser independentes dele (o ZeroGPU derruba o contexto CUDA
        # entre chamadas, e um cache apontando para tensores CUDA fica inválido).
        self.blocks = nn.ModuleList(
            copy.deepcopy(getattr(base, name)) for name in order[: indices[-1] + 1]
        )
        self.taps = set(indices)

        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)

    def forward(self, img):
        """img: tensor (3, H, W) em [0, 1]."""
        x = self.transform(img).unsqueeze(0)
        activations = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.taps:
                activations.append(x)
        return activations


def _objective(activations, mode):
    """L2 é o objetivo do notebook original; mean é o do tutorial em TensorFlow."""
    if mode == "l2":
        return sum(a.pow(2).sum() / 2 for a in activations)
    return sum(a.mean() for a in activations)


def _gradient(model, img, objective):
    img = img.detach().requires_grad_(True)
    loss = _objective(model(img), objective)
    (grad,) = torch.autograd.grad(loss, img)
    return grad, loss.item()


def _tiled_gradient(model, img, objective, tile_size):
    """Gradiente por ladrilhos, para imagens que não cabem na memória de uma vez.

    Este caminho não existe no original: use tile_size grande o bastante para
    não disparar se o objetivo for fidelidade máxima.
    """
    _, height, width = img.shape
    shift_y, shift_x = np.random.randint(-tile_size // 2, tile_size // 2 + 1, size=2)

    rolled = torch.roll(img, shifts=(int(shift_y), int(shift_x)), dims=(1, 2))
    rolled = rolled.detach().requires_grad_(True)

    total_loss = 0.0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = rolled[:, y : y + tile_size, x : x + tile_size]
            if min(tile.shape[1], tile.shape[2]) < 96:
                continue
            loss = _objective(model(tile), objective)
            loss.backward()
            total_loss += loss.item()

    grad = torch.roll(rolled.grad, shifts=(-int(shift_y), -int(shift_x)), dims=(1, 2))
    return grad, total_loss


def _zoom(array, factor_y, factor_x):
    """Interpolação linear do scipy, a mesma que o notebook original usa."""
    return nd.zoom(array, (1, factor_y, factor_x), order=1)


def _make_step(model, img, objective, step_size, jitter, tile_size):
    """Um passo de subida de gradiente, equivalente ao make_step original."""
    if jitter:
        ox, oy = np.random.randint(-jitter, jitter + 1, size=2)
        img = torch.roll(img, shifts=(int(oy), int(ox)), dims=(1, 2))

    if max(img.shape[1], img.shape[2]) <= tile_size:
        grad, loss = _gradient(model, img, objective)
    else:
        grad, loss = _tiled_gradient(model, img, objective, tile_size)

    # Passo normalizado pela média absoluta do gradiente, como no Caffe.
    # step_size é dado em unidades de pixel 0-255; aqui a imagem vive em [0, 1].
    img = img + (step_size / 255.0) / (grad.abs().mean() + 1e-8) * grad

    if jitter:
        img = torch.roll(img, shifts=(-int(oy), -int(ox)), dims=(1, 2))

    # Equivale ao clip do original para [-média, 255-média].
    return img.clamp(0, 1).detach(), loss


def dream(
    image,
    layers=(DEFAULT_LAYER,),
    model=DEFAULT_MODEL,
    iterations=DEFAULT_ITERATIONS,
    step_size=DEFAULT_STEP_SIZE,
    octaves=DEFAULT_OCTAVES,
    octave_scale=DEFAULT_OCTAVE_SCALE,
    jitter=DEFAULT_JITTER,
    max_dim=1024,
    tile_size=512,
    objective="l2",
    seed=None,
    device=None,
    on_step=None,
):
    """Aplica DeepDream numa PIL.Image e devolve uma PIL.Image."""
    if seed is not None:
        np.random.seed(seed)

    device = pick_device(device)
    extractor = FeatureExtractor(model, layers, device)

    image = image.convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)

    base = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    # Pirâmide: da imagem menor para a maior, como no notebook original.
    pyramid = [base]
    for _ in range(octaves - 1):
        pyramid.append(_zoom(pyramid[-1], 1.0 / octave_scale, 1.0 / octave_scale))
    pyramid.reverse()

    total_steps = octaves * iterations
    done = 0
    detail = np.zeros_like(pyramid[0])

    for index, octave_base in enumerate(pyramid):
        height, width = octave_base.shape[-2:]
        if index > 0:
            # O detalhe acumulado sobe de escala e é reinjetado na octave maior.
            prev_h, prev_w = detail.shape[-2:]
            detail = _zoom(detail, height / prev_h, width / prev_w)

        img = torch.from_numpy(octave_base + detail).to(device)

        for _ in range(iterations):
            img, loss = _make_step(
                extractor, img, objective, step_size, jitter, tile_size
            )
            done += 1
            if on_step:
                on_step(done, total_steps, loss)

        detail = img.cpu().numpy() - octave_base

    array = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def main():
    parser = argparse.ArgumentParser(description="DeepDream clássico local em PyTorch.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "-l", "--layers", default=DEFAULT_LAYER,
        help="Camadas separadas por vírgula, ex.: inception_4c/output",
    )
    parser.add_argument("-m", "--model", choices=list(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("-n", "--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--octaves", type=int, default=DEFAULT_OCTAVES)
    parser.add_argument("--octave-scale", type=float, default=DEFAULT_OCTAVE_SCALE)
    parser.add_argument("--jitter", type=int, default=DEFAULT_JITTER)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--objective", choices=["l2", "mean"], default="l2")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_dream.png")

    def report(done, total, loss):
        print(f"\rPasso {done}/{total} — loss {loss:.3e}", end="", flush=True)

    result = dream(
        Image.open(args.input),
        layers=[s for s in args.layers.split(",") if s.strip()],
        model=args.model,
        iterations=args.iterations,
        step_size=args.step_size,
        octaves=args.octaves,
        octave_scale=args.octave_scale,
        jitter=args.jitter,
        max_dim=args.max_dim,
        tile_size=args.tile_size,
        objective=args.objective,
        seed=args.seed,
        device=args.device,
        on_step=report,
    )
    print()

    result.save(output)
    print(f"Salvo em {output}")


if __name__ == "__main__":
    main()
