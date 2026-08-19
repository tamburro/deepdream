"""Interface web do DeepDream clássico.

Roda local (`python app.py`) ou hospedado. Configuração por variáveis de ambiente:
  DEEPDREAM_DEVICE   mps | cuda | cpu   (padrão: melhor disponível)
  DEEPDREAM_MAX_DIM  teto de resolução  (padrão: 1536, ou 768 se estiver em CPU)
  DEEPDREAM_SHARE    1 para gerar link público temporário do Gradio
  PORT               porta do servidor  (padrão: 7860)
"""

import os
import tempfile

import gradio as gr

from deepdream import (
    DEFAULT_ITERATIONS,
    DEFAULT_JITTER,
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

# Em CPU (o caso de qualquer hospedagem gratuita) um teto alto vira espera longa.
DEFAULT_MAX_DIM_CAP = 768 if DEVICE.type == "cpu" else 1536
MAX_DIM_CAP = int(os.environ.get("DEEPDREAM_MAX_DIM", DEFAULT_MAX_DIM_CAP))

LAYER_CHOICES = [f"{b}/output" for b in INCEPTION_BLOCKS]

MODEL_LABELS = {
    "bvlc": "bvlc — ImageNet, pesos Caffe originais (o DeepDream de 2015)",
    "places365": "places365 — cenários: templos, arcos, arquitetura",
    "places205": "places205 — cenários, versão antiga",
    "torchvision": "torchvision — outro treinamento, com batchnorm: mais suave",
}

# Combinações que valem a pena ter a um clique de distância.
PRESETS = {
    "Clássico 2015": dict(model="bvlc", layers=["inception_4c/output"], iterations=10,
                          step_size=1.5, octaves=4, octave_scale=1.4),
    "Puppy slugs (forte)": dict(model="bvlc", layers=["inception_4c/output"], iterations=25,
                                step_size=2.0, octaves=5, octave_scale=1.4),
    "Criaturas híbridas": dict(model="bvlc", layers=["inception_4d/output"], iterations=20,
                               step_size=1.5, octaves=4, octave_scale=1.4),
    "Quimeras bizarras": dict(model="bvlc", layers=["inception_5b/output"], iterations=40,
                              step_size=2.0, octaves=4, octave_scale=1.4),
    "Só textura": dict(model="bvlc", layers=["inception_3b/output"], iterations=10,
                       step_size=1.5, octaves=4, octave_scale=1.4),
    "Templos e arcos": dict(model="places365", layers=["inception_4d/output"], iterations=20,
                            step_size=1.5, octaves=4, octave_scale=1.4),
    "Suave (torchvision)": dict(model="torchvision", layers=["inception_4c/output"], iterations=10,
                                step_size=1.5, octaves=4, octave_scale=1.4),
}

LAYER_GUIDE = """
| Camada | O que costuma brotar |
| --- | --- |
| `inception_3a` / `3b` | Traços, bordas e texturas simples |
| `inception_4a` / `4b` | Espirais, olhos soltos, padrões repetidos |
| **`inception_4c`** | **Focinhos de cachorro, pelos, olhos isolados — o visual clássico** |
| `inception_4d` | Animais de corpo inteiro distorcidos, pássaros, híbridos |
| `inception_4e` | Criaturas maiores, arquitetura, formas compostas |
| `inception_5a` | Peixes, anfíbios, sapos, olhos reptilianos |
| `inception_5b` | Quimeras: macacos, lagartos, cobras — o mais bizarro |

As camadas profundas (`5a`, `5b`) precisam de mais iterações que o padrão 10
para render bem. O guia vale para os modelos treinados na ImageNet (`bvlc`).
"""


def apply_preset(name):
    p = PRESETS[name]
    return (
        MODEL_LABELS[p["model"]], p["layers"], p["iterations"],
        p["step_size"], p["octaves"], p["octave_scale"],
    )


def run(
    image, model, layers, iterations, step_size, octaves, octave_scale,
    jitter, max_dim, objective, seed, progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Escolha uma imagem.")
    if not layers:
        raise gr.Error("Selecione ao menos uma camada.")

    def report(done, total, loss):
        progress(done / total, desc=f"Passo {done}/{total}")

    result = dream(
        image,
        layers=layers,
        model=model.split(" — ")[0],
        iterations=int(iterations),
        step_size=step_size,
        octaves=int(octaves),
        octave_scale=octave_scale,
        jitter=int(jitter),
        max_dim=min(int(max_dim), MAX_DIM_CAP),
        objective=objective,
        seed=int(seed) if seed is not None else None,
        device=DEVICE,
        on_step=report,
    )

    path = os.path.join(tempfile.mkdtemp(), "deepdream.png")
    result.save(path)
    return (image, result), path


with gr.Blocks(title="DeepDream") as demo:
    gr.Markdown(
        f"# DeepDream clássico\n"
        f"Reprodução do DeepDream original do Google (2015), com os pesos "
        f"`bvlc_googlenet` do Caffe. Rodando em `{DEVICE.type}`."
    )

    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(type="pil", label="Imagem", height=320)
            preset = gr.Radio(
                choices=list(PRESETS), value="Clássico 2015", label="Preset"
            )
            run_button = gr.Button("Sonhar", variant="primary", size="lg")

            with gr.Accordion("Ajustes", open=False):
                model = gr.Dropdown(
                    choices=[MODEL_LABELS[name] for name in MODELS],
                    value=MODEL_LABELS[DEFAULT_MODEL], label="Modelo",
                )
                layers = gr.CheckboxGroup(
                    choices=LAYER_CHOICES, value=["inception_4c/output"], label="Camadas",
                )
                iterations = gr.Slider(1, 100, value=DEFAULT_ITERATIONS, step=1,
                                       label="Iterações por octave")
                step_size = gr.Slider(0.1, 6.0, value=DEFAULT_STEP_SIZE, step=0.1,
                                      label="Tamanho do passo")
                octaves = gr.Slider(1, 8, value=DEFAULT_OCTAVES, step=1, label="Octaves")
                octave_scale = gr.Slider(1.1, 2.0, value=DEFAULT_OCTAVE_SCALE, step=0.05,
                                         label="Escala por octave")
                jitter = gr.Slider(0, 64, value=DEFAULT_JITTER, step=1, label="Jitter")
                max_dim = gr.Slider(256, MAX_DIM_CAP, value=min(1024, MAX_DIM_CAP), step=128,
                                    label="Dimensão máxima")
                objective = gr.Radio(["l2", "mean"], value="l2", label="Objetivo",
                                     info="l2 é o do notebook original.")
                seed = gr.Number(value=None, label="Seed", precision=0,
                                 info="Só é reprodutível de fato em CPU.")

            with gr.Accordion("Guia de camadas", open=False):
                gr.Markdown(LAYER_GUIDE)

        with gr.Column(scale=3):
            comparison = gr.ImageSlider(label="Antes / depois", height=520)
            download = gr.File(label="Baixar PNG")

    if os.path.exists("exemplo.jpg"):
        gr.Examples(examples=[["exemplo.jpg"]], inputs=[image], label="Exemplo")

    preset.change(
        apply_preset, inputs=[preset],
        outputs=[model, layers, iterations, step_size, octaves, octave_scale],
    )
    run_button.click(
        run,
        inputs=[image, model, layers, iterations, step_size, octaves, octave_scale,
                jitter, max_dim, objective, seed],
        outputs=[comparison, download],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("DEEPDREAM_SHARE") == "1",
    )
