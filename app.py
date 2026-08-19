"""Interface web do Dream Canvas.

Roda local (`python app.py`) ou hospedado. Configuração por variáveis de ambiente:
  DEEPDREAM_DEVICE   mps | cuda | cpu   (padrão: melhor disponível)
  DEEPDREAM_MAX_DIM  teto de resolução  (padrão: 1536, ou 768 se estiver em CPU)
  DEEPDREAM_SHARE    1 para gerar link público temporário do Gradio
  PORT               porta do servidor  (padrão: 7860)
  DEEPDREAM_OUTPUT_DIR       onde salvar as saídas (padrão: $TMPDIR/deepdream)
  DEEPDREAM_OUTPUT_MAX_AGE_H horas até uma saída ser apagada (padrão: 24)
"""

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

# No ZeroGPU o pacote spaces precisa ser importado antes do torch. Fora dele,
# o pacote não existe e o decorador vira um no-op.
try:
    import spaces

    ZEROGPU = True
except ImportError:
    ZEROGPU = False

    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn=None, **kwargs):
            return fn if fn is not None else (lambda f: f)

import gradio as gr
from PIL import ImageDraw

try:
    import video as video_module

    VIDEO_OK = not ZEROGPU and shutil.which("ffmpeg") is not None
except ImportError:
    video_module = None
    VIDEO_OK = False
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

# Uma pasta só, previsível, em vez de um mkdtemp novo por geração — assim dá
# para achar os arquivos, e a limpeza acontece sozinha.
OUTPUT_DIR = Path(
    os.environ.get("DEEPDREAM_OUTPUT_DIR", Path(tempfile.gettempdir()) / "deepdream")
)
OUTPUT_MAX_AGE = float(os.environ.get("DEEPDREAM_OUTPUT_MAX_AGE_H", 24)) * 3600


def output_path(suffix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return str(OUTPUT_DIR / f"{stamp}-{uuid.uuid4().hex[:6]}{suffix}")


def prune_outputs():
    """Apaga saídas antigas na inicialização, para a pasta não crescer sem fim."""
    if not OUTPUT_DIR.exists():
        return
    cutoff = time.time() - OUTPUT_MAX_AGE
    for item in OUTPUT_DIR.iterdir():
        if item.is_file() and item.stat().st_mtime < cutoff:
            item.unlink(missing_ok=True)


prune_outputs()

FORCED_DEVICE = os.environ.get("DEEPDREAM_DEVICE")

# No ZeroGPU a GPU só existe dentro da função decorada, então o device é
# resolvido a cada execução, e não uma vez na importação.
def current_device():
    return pick_device(FORCED_DEVICE)


# Em CPU pura um teto alto vira espera longa; no ZeroGPU há GPU de sobra.
_startup_device = current_device()
DEFAULT_MAX_DIM_CAP = 768 if _startup_device.type == "cpu" and not ZEROGPU else 1536
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


@spaces.GPU(duration=120)
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
        device=current_device(),
        on_step=report,
    )

    path = output_path(".png")
    result.save(path)
    return (image, result), path


def run_video(
    path, model, layers, iterations, step_size, octaves, blend,
    max_dim, use_flow, start, duration, progress=gr.Progress(),
):
    if not path:
        raise gr.Error("Escolha um vídeo.")

    def report(index, total):
        if total:
            progress(index / total, desc=f"Quadro {index}/{total}")
        else:
            progress(0, desc=f"Quadro {index}")

    output = output_path(".mp4")
    try:
        return video_module.process(
            path, output,
            layers=layers,
            model=model.split(" — ")[0],
            iterations=int(iterations),
            step_size=step_size,
            octaves=int(octaves),
            max_dim=int(max_dim),
            blend=blend,
            flow=use_flow,
            start=start or None,
            duration=duration or None,
            device=current_device(),
            on_frame=report,
        )
    except RuntimeError as error:
        raise gr.Error(str(error))


def apply_video_preset(name):
    p = PRESETS[name]
    return MODEL_LABELS[p["model"]], p["layers"], p["step_size"]


GRADIO_CACHE = Path(os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "gradio"))


def _folder_size(folder):
    if not folder.exists():
        return 0, 0
    files = [f for f in folder.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def _human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit != "GB" else f"{size:.1f} {unit}"
        size /= 1024


def cache_usage():
    ours_count, ours_size = _folder_size(OUTPUT_DIR)
    gradio_count, gradio_size = _folder_size(GRADIO_CACHE)
    return (
        f"**Saídas geradas:** {ours_count} arquivos, {_human(ours_size)} — "
        f"`{OUTPUT_DIR}`<br>"
        f"**Cache do Gradio:** {gradio_count} arquivos, {_human(gradio_size)} — "
        f"`{GRADIO_CACHE}`"
    )


def clear_cache(include_gradio):
    """Apaga as saídas geradas. Nunca toca nos pesos dos modelos em ~/.cache/torch,
    porque apagá-los só forçaria baixar tudo de novo."""
    freed = 0
    folders = [OUTPUT_DIR] + ([GRADIO_CACHE] if include_gradio else [])

    for folder in folders:
        if not folder.exists():
            continue
        for item in folder.rglob("*"):
            if item.is_file():
                try:
                    freed += item.stat().st_size
                    item.unlink()
                except OSError:
                    pass

    gr.Info(f"Liberado: {_human(freed)}")
    return cache_usage()


def mark_center(image, evt: gr.SelectData):
    """Desenha uma mira no ponto clicado e devolve o centro normalizado."""
    if image is None:
        return None, (0.5, 0.5), "Centro: meio da imagem"

    x, y = evt.index
    center = (x / image.width, y / image.height)

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    radius = max(8, min(marked.width, marked.height) // 40)
    for width, color in ((4, "black"), (2, "white")):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=color, width=width)
        draw.line([x - radius * 2, y, x + radius * 2, y], fill=color, width=width)
        draw.line([x, y - radius * 2, x, y + radius * 2], fill=color, width=width)

    return marked, center, f"Centro: {center[0]:.0%} × {center[1]:.0%}"


def run_zoom(
    image, center, model, layers, iterations, step_size, octaves,
    duration, fps, zoom, max_dim, progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Escolha uma imagem.")

    def report(index, total):
        progress(index / total, desc=f"Quadro {index}/{total}")

    output = output_path("-zoom.mp4")
    try:
        return video_module.zoom_video(
            image, output,
            center=center or (0.5, 0.5),
            duration=duration,
            fps=int(fps),
            zoom=zoom,
            layers=layers,
            model=model.split(" — ")[0],
            iterations=int(iterations),
            step_size=step_size,
            octaves=int(octaves),
            max_dim=int(max_dim),
            device=current_device(),
            on_frame=report,
        )
    except RuntimeError as error:
        raise gr.Error(str(error))


with gr.Blocks(title="Dream Canvas") as demo:
    gr.Markdown(
        f"# Dream Canvas\n"
        f"Reprodução do DeepDream original do Google (2015), com os pesos "
        f"`bvlc_googlenet` do Caffe. Rodando em "
        f"`{'ZeroGPU' if ZEROGPU else _startup_device.type}`."
    )
    gr.Markdown(
        f"<sub>As saídas ficam em <code>{OUTPUT_DIR}</code> e são apagadas "
        f"depois de {int(OUTPUT_MAX_AGE / 3600)} h. Baixe o que quiser guardar.</sub>"
    )

    with gr.Tab("Imagem"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(type="pil", label="Imagem", height=320)
                if VIDEO_OK:
                    gr.Markdown(
                        "<sub>Só imagem aqui. Para vídeo, use a aba "
                        "**Vídeo** acima.</sub>"
                    )
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

    if VIDEO_OK:
        with gr.Tab("Vídeo"):
            gr.Markdown(
                "Cada quadro sonhado realimenta o próximo, deformado pelo fluxo "
                "óptico — sem isso o vídeo pisca violentamente. Conte com uns "
                "**0,7 s por quadro** a 640px; comece recortando poucos segundos."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    video_in = gr.Video(label="Vídeo", height=320)
                    video_preset = gr.Radio(
                        choices=list(PRESETS), value="Clássico 2015", label="Preset"
                    )
                    video_button = gr.Button("Sonhar o vídeo", variant="primary", size="lg")

                    with gr.Row():
                        video_start = gr.Number(value=0, label="Início (s)", precision=1)
                        video_duration = gr.Number(value=3, label="Duração (s)", precision=1,
                                                   info="0 processa até o fim.")

                    with gr.Accordion("Ajustes", open=False):
                        video_model = gr.Dropdown(
                            choices=[MODEL_LABELS[name] for name in MODELS],
                            value=MODEL_LABELS[DEFAULT_MODEL], label="Modelo",
                        )
                        video_layers = gr.CheckboxGroup(
                            choices=LAYER_CHOICES, value=["inception_4c/output"],
                            label="Camadas",
                        )
                        video_iterations = gr.Slider(
                            1, 40, value=video_module.DEFAULT_ITERATIONS, step=1,
                            label="Iterações por quadro",
                        )
                        video_step = gr.Slider(0.1, 6.0, value=DEFAULT_STEP_SIZE, step=0.1,
                                               label="Tamanho do passo")
                        video_octaves = gr.Slider(
                            1, 6, value=video_module.DEFAULT_OCTAVES, step=1, label="Octaves",
                        )
                        video_blend = gr.Slider(
                            0.0, 0.95, value=video_module.DEFAULT_BLEND, step=0.05,
                            label="Realimentação",
                            info="Quanto do quadro anterior volta. Mais alto: "
                                 "mais estável, mais rastro.",
                        )
                        video_max_dim = gr.Slider(
                            256, 1280, value=video_module.DEFAULT_MAX_DIM, step=64,
                            label="Dimensão máxima",
                        )
                        video_flow = gr.Checkbox(
                            value=True, label="Fluxo óptico",
                            info="Desligar acelera, mas o padrão descola do movimento.",
                        )

                with gr.Column(scale=3):
                    video_out = gr.Video(label="Resultado", height=520)

            video_preset.change(
                apply_video_preset, inputs=[video_preset],
                outputs=[video_model, video_layers, video_step],
            )
            video_button.click(
                run_video,
                inputs=[video_in, video_model, video_layers, video_iterations,
                        video_step, video_octaves, video_blend, video_max_dim,
                        video_flow, video_start, video_duration],
                outputs=video_out,
            )

    if VIDEO_OK:
        with gr.Tab("Zoom infinito"):
            gr.Markdown(
                "Sonha um quadro, aproxima um pouco em direção ao ponto escolhido, "
                "repete — detalhe novo nasce no centro sem parar. "
                "**Clique na imagem** para escolher para onde o zoom vai."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    zoom_image = gr.Image(type="pil", label="Imagem (clique para mirar)",
                                          height=300)
                    zoom_center = gr.State((0.5, 0.5))
                    zoom_center_label = gr.Markdown("Centro: meio da imagem")
                    zoom_marked = gr.Image(label="Ponto escolhido", height=220,
                                           interactive=False)

                    zoom_preset = gr.Radio(
                        choices=list(PRESETS), value="Clássico 2015", label="Preset"
                    )
                    zoom_duration = gr.Slider(1, 30, value=5, step=1, label="Duração (s)")
                    zoom_speed = gr.Slider(
                        0.005, 0.08, value=video_module.DEFAULT_ZOOM, step=0.005,
                        label="Velocidade do zoom",
                        info="Quanto a imagem avança por quadro.",
                    )
                    zoom_button = gr.Button("Gerar zoom", variant="primary", size="lg")

                    with gr.Accordion("Ajustes", open=False):
                        zoom_model = gr.Dropdown(
                            choices=[MODEL_LABELS[name] for name in MODELS],
                            value=MODEL_LABELS[DEFAULT_MODEL], label="Modelo",
                        )
                        zoom_layers = gr.CheckboxGroup(
                            choices=LAYER_CHOICES, value=["inception_4c/output"],
                            label="Camadas",
                        )
                        zoom_iterations = gr.Slider(
                            1, 40, value=video_module.DEFAULT_ITERATIONS, step=1,
                            label="Iterações por quadro",
                        )
                        zoom_step = gr.Slider(0.1, 6.0, value=DEFAULT_STEP_SIZE, step=0.1,
                                              label="Tamanho do passo")
                        zoom_octaves = gr.Slider(
                            1, 6, value=video_module.DEFAULT_OCTAVES, step=1,
                            label="Octaves",
                        )
                        zoom_fps = gr.Slider(
                            10, 30, value=video_module.DEFAULT_ZOOM_FPS, step=1, label="FPS",
                        )
                        zoom_max_dim = gr.Slider(
                            256, 1024, value=512, step=64, label="Dimensão máxima",
                        )

                with gr.Column(scale=3):
                    zoom_out = gr.Video(label="Resultado", height=520)
                    gr.Markdown(
                        "<sub>Conte ~0,4 s por quadro a 512px. 5 s a 20 fps são "
                        "100 quadros, ou seja, cerca de 40 s de espera.</sub>"
                    )

            zoom_preset.change(
                apply_video_preset, inputs=[zoom_preset],
                outputs=[zoom_model, zoom_layers, zoom_step],
            )
            zoom_image.select(
                mark_center, inputs=[zoom_image],
                outputs=[zoom_marked, zoom_center, zoom_center_label],
            )
            zoom_button.click(
                run_zoom,
                inputs=[zoom_image, zoom_center, zoom_model, zoom_layers,
                        zoom_iterations, zoom_step, zoom_octaves, zoom_duration,
                        zoom_fps, zoom_speed, zoom_max_dim],
                outputs=zoom_out,
            )

    with gr.Accordion("Manutenção", open=False):
        usage = gr.Markdown(cache_usage())
        include_gradio = gr.Checkbox(
            value=False, label="Incluir o cache do Gradio",
            info="Também apaga uploads e miniaturas. Resultados já exibidos "
                 "na tela podem deixar de carregar até você gerar de novo.",
        )
        with gr.Row():
            refresh_button = gr.Button("Atualizar", size="sm")
            clear_button = gr.Button("Limpar cache", variant="stop", size="sm")
        gr.Markdown(
            "<sub>Os pesos dos modelos (`~/.cache/torch`) nunca são apagados — "
            "removê-los só forçaria baixar tudo outra vez.</sub>"
        )

        refresh_button.click(cache_usage, outputs=usage)
        clear_button.click(clear_cache, inputs=[include_gradio], outputs=usage)

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
