"""API HTTP do DeepDream, para integrações externas (ex.: o plugin do Figma).

Separada da interface Gradio de propósito: aqui o contrato é simples e estável
— manda PNG em base64, recebe PNG em base64.

    .venv/bin/python server.py

O CORS é liberado porque o iframe de um plugin do Figma tem origem `null`, e
porque este servidor é feito para escutar só em localhost.
"""

import base64
import io
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from deepdream import (
    DEFAULT_ITERATIONS,
    DEFAULT_JITTER,
    DEFAULT_LAYER,
    DEFAULT_MODEL,
    DEFAULT_OCTAVE_SCALE,
    DEFAULT_OCTAVES,
    DEFAULT_STEP_SIZE,
    INCEPTION_BLOCKS,
    MODELS,
    dream,
    pick_device,
)

DEVICE = pick_device(os.environ.get("DEEPDREAM_DEVICE"))

api = FastAPI(title="DeepDream")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DreamRequest(BaseModel):
    image: str  # PNG em base64, com ou sem o prefixo data:
    layers: list[str] = [DEFAULT_LAYER]
    model: str = DEFAULT_MODEL
    iterations: int = DEFAULT_ITERATIONS
    step_size: float = DEFAULT_STEP_SIZE
    octaves: int = DEFAULT_OCTAVES
    octave_scale: float = DEFAULT_OCTAVE_SCALE
    jitter: int = DEFAULT_JITTER
    max_dim: int = 1024
    objective: str = "l2"
    seed: int | None = None


@api.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE.type,
        "models": list(MODELS),
        "layers": [f"{b}/output" for b in INCEPTION_BLOCKS],
    }


@api.post("/dream")
def post_dream(request: DreamRequest):
    payload = request.image.split(",", 1)[-1]
    try:
        source = Image.open(io.BytesIO(base64.b64decode(payload)))
    except Exception:
        raise HTTPException(400, "Não consegui decodificar a imagem enviada.")

    if request.model not in MODELS:
        raise HTTPException(400, f"Modelo desconhecido: {request.model}")

    try:
        result = dream(
            source,
            layers=request.layers,
            model=request.model,
            iterations=request.iterations,
            step_size=request.step_size,
            octaves=request.octaves,
            octave_scale=request.octave_scale,
            jitter=request.jitter,
            max_dim=request.max_dim,
            objective=request.objective,
            seed=request.seed,
            device=DEVICE,
        )
    except ValueError as error:
        raise HTTPException(400, str(error))

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return {
        "image": base64.b64encode(buffer.getvalue()).decode(),
        "width": result.width,
        "height": result.height,
    }


if __name__ == "__main__":
    uvicorn.run(
        api,
        host="127.0.0.1",
        port=int(os.environ.get("DEEPDREAM_API_PORT", 8000)),
    )
