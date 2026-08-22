"""Sonhar em direção a um texto, usando CLIP.

Diferente do DeepDream clássico: em vez de amplificar as ativações de uma
camada, a imagem é otimizada para que o CLIP a considere parecida com uma
descrição escrita.

O truque que faz isso funcionar são os **recortes aleatórios**. Otimizar a
imagem inteira contra o texto produz ruído adversarial — padrões que o CLIP
reconhece e nós não. Pontuando dezenas de recortes de tamanhos variados a cada
passo, o gradiente passa a exigir que a descrição valha em muitas escalas e
posições, e o resultado vira forma de verdade.

Dependência à parte: `pip install -r requirements-clip.txt`.
"""

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

# O tag "openai" foi treinado com QuickGELU; usar a variante errada muda a
# ativação e degrada o resultado silenciosamente (o open_clip só avisa).
DEFAULT_CLIP = "ViT-B-32-quickgelu"
DEFAULT_PRETRAINED = "openai"
DEFAULT_CUTOUTS = 16
CLIP_SIZE = 224

# Normalização própria do CLIP — não é a da ImageNet.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_CACHE = {}


def pick_device(preferred=None):
    """Device para o CLIP. Evita MPS de propósito.

    Medido num M4: o forward do ViT em MPS leva 0,08 s, mas o **backward** leva
    39,5 s — contra 0,38 s na CPU. Cem vezes mais lento, de forma consistente e
    já com o backward aquecido. É limitação do backend MPS, e como o DeepDream
    guiado por texto é feito de backwards, a CPU é a escolha certa aqui.
    """
    import torch as _torch

    if preferred:
        return _torch.device(preferred)
    if _torch.cuda.is_available():
        return _torch.device("cuda")
    return _torch.device("cpu")


def load_clip(device, model_name=DEFAULT_CLIP, pretrained=DEFAULT_PRETRAINED):
    key = (model_name, pretrained, str(device))
    if key not in _CACHE:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (model, open_clip.get_tokenizer(model_name))
    return _CACHE[key]


class TextScorer(nn.Module):
    """Pontua o quanto a imagem se parece com o texto, na visão do CLIP."""

    def __init__(self, text, device, model_name=DEFAULT_CLIP,
                 pretrained=DEFAULT_PRETRAINED, cutouts=DEFAULT_CUTOUTS):
        super().__init__()
        self.model, tokenizer = load_clip(device, model_name, pretrained)
        self.cutouts = cutouts
        self.device = device

        with torch.no_grad():
            tokens = tokenizer([text]).to(device)
            target = self.model.encode_text(tokens).float()
        self.target = F.normalize(target, dim=-1)

        self.register_buffer("mean", torch.tensor(CLIP_MEAN, device=device).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD, device=device).view(3, 1, 1))

    def _crops(self, img):
        """Recortes quadrados de tamanhos variados, todos levados a 224."""
        _, height, width = img.shape
        smallest = min(height, width)
        crops = []
        for _ in range(self.cutouts):
            # Entre 30% e 100% do menor lado: cobre desde detalhe até composição.
            size = int(np.random.uniform(0.3, 1.0) * smallest)
            size = max(CLIP_SIZE // 4, size)
            y = np.random.randint(0, max(1, height - size + 1))
            x = np.random.randint(0, max(1, width - size + 1))
            crop = img[:, y : y + size, x : x + size].unsqueeze(0)
            crops.append(
                F.interpolate(crop, size=(CLIP_SIZE, CLIP_SIZE),
                              mode="bilinear", align_corners=False)
            )
        return torch.cat(crops)

    def forward(self, img):
        """img: tensor (3, H, W) em [0, 1]. Devolve a similaridade média."""
        crops = (self._crops(img) - self.mean) / self.std
        features = F.normalize(self.model.encode_image(crops).float(), dim=-1)
        return (features @ self.target.t()).mean()
