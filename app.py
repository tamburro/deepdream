"""Interface do Dream Canvas.

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

import theme as dc_theme

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
GRADIO_CACHE = Path(
    os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "gradio")
)


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


_startup_device = current_device()
DEFAULT_MAX_DIM_CAP = 768 if _startup_device.type == "cpu" and not ZEROGPU else 1536
MAX_DIM_CAP = int(os.environ.get("DEEPDREAM_MAX_DIM", DEFAULT_MAX_DIM_CAP))

# Tuplas (rótulo, valor): o callback recebe a chave, sem precisar fatiar string.
LAYER_CHOICES = [(b.replace("inception_", ""), f"{b}/output") for b in INCEPTION_BLOCKS]

MODEL_CHOICES = [
    ("Clássico — ImageNet, pesos originais de 2015", "bvlc"),
    ("Cenários — templos, arcos, arquitetura", "places365"),
    ("Cenários (versão antiga)", "places205"),
    ("Suave — outro treinamento, formas mais discretas", "torchvision"),
]

MODE_CHOICES = [("Clássico", "l2"), ("Suave", "mean")]

PRESETS = {
    "Clássico 2015": dict(model="bvlc", layers=["inception_4c/output"], iterations=10,
                          step_size=1.5, octaves=4, octave_scale=1.4),
    "Focinhos por toda parte": dict(model="bvlc", layers=["inception_4c/output"], iterations=25,
                                    step_size=2.0, octaves=5, octave_scale=1.4),
    "Criaturas híbridas": dict(model="bvlc", layers=["inception_4d/output"], iterations=20,
                               step_size=1.5, octaves=4, octave_scale=1.4),
    "Quimeras": dict(model="bvlc", layers=["inception_5b/output"], iterations=40,
                     step_size=2.0, octaves=4, octave_scale=1.4),
    "Só textura": dict(model="bvlc", layers=["inception_3b/output"], iterations=10,
                       step_size=1.5, octaves=4, octave_scale=1.4),
    "Templos e arcos": dict(model="places365", layers=["inception_4d/output"], iterations=20,
                            step_size=1.5, octaves=4, octave_scale=1.4),
    "Discreto": dict(model="torchvision", layers=["inception_4c/output"], iterations=10,
                     step_size=1.5, octaves=4, octave_scale=1.4),
}

LAYER_GUIDE = """
Cada camada da rede aprendeu a reconhecer coisas diferentes. Escolher uma
determina o que vai brotar da sua imagem.

| Camada | O que brota |
| --- | --- |
| `3a` `3b` | Traços, bordas, textura pura |
| `4a` `4b` | Espirais, olhos soltos, padrões repetidos |
| `4c` | Focinhos, pelo, olhos isolados — o visual clássico de 2015 |
| `4d` | Animais inteiros e distorcidos, pássaros, híbridos |
| `4e` | Criaturas maiores, arquitetura, formas compostas |
| `5a` | Peixes, anfíbios, olhos reptilianos |
| `5b` | Quimeras: macacos, lagartos, cobras |

As camadas fundas (`5a`, `5b`) precisam de 30 a 50 iterações para render bem.
O guia vale para os modelos treinados na ImageNet — o Clássico e o Suave.
"""


# ---------------------------------------------------------------- lógica

def apply_preset(name):
    p = PRESETS[name]
    return (p["model"], p["layers"], p["iterations"],
            p["step_size"], p["octaves"], p["octave_scale"])


def apply_motion_preset(name):
    """Presets em vídeo e zoom mexem só no estilo, não no ritmo."""
    p = PRESETS[name]
    return p["model"], p["layers"], p["step_size"]


@spaces.GPU(duration=120)
def run(image, model, layers, iterations, step_size, octaves, octave_scale,
        jitter, max_dim, mode, seed, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Escolha uma imagem.")
    if not layers:
        raise gr.Error("Escolha ao menos uma camada.")

    def report(done, total, loss):
        progress(done / total, desc=f"Passo {done}/{total}")

    result = dream(
        image, layers=layers, model=model,
        iterations=int(iterations), step_size=step_size,
        octaves=int(octaves), octave_scale=octave_scale,
        jitter=int(jitter), max_dim=min(int(max_dim), MAX_DIM_CAP),
        objective=mode, seed=int(seed) if seed is not None else None,
        device=current_device(), on_step=report,
    )

    path = output_path(".png")
    result.save(path)
    return (image, result), path


def run_video(path, model, layers, iterations, step_size, octaves, blend,
              max_dim, use_flow, start, duration, progress=gr.Progress()):
    if not path:
        raise gr.Error("Escolha um vídeo.")

    def report(index, total):
        if total:
            progress(index / total, desc=f"Quadro {index}/{total}")
        else:
            progress(0, desc=f"Quadro {index}")

    try:
        return video_module.process(
            path, output_path(".mp4"), layers=layers, model=model,
            iterations=int(iterations), step_size=step_size, octaves=int(octaves),
            max_dim=int(max_dim), blend=blend, flow=use_flow,
            start=start or None, duration=duration or None,
            device=current_device(), on_frame=report,
        )
    except RuntimeError as error:
        raise gr.Error(str(error))


def mark_center(image, evt: gr.SelectData):
    """Desenha uma mira no ponto clicado e devolve o centro normalizado."""
    if image is None:
        return None, (0.5, 0.5), "Mirando no meio da imagem."

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

    return marked, center, f"Mirando em {center[0]:.0%} × {center[1]:.0%}."


def run_zoom(image, center, model, layers, iterations, step_size, octaves,
             duration, fps, zoom, max_dim, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Escolha uma imagem.")

    def report(index, total):
        progress(index / total, desc=f"Quadro {index}/{total}")

    try:
        return video_module.zoom_video(
            image, output_path("-zoom.mp4"), center=center or (0.5, 0.5),
            duration=duration, fps=int(fps), zoom=zoom, layers=layers,
            model=model, iterations=int(iterations), step_size=step_size,
            octaves=int(octaves), max_dim=int(max_dim),
            device=current_device(), on_frame=report,
        )
    except RuntimeError as error:
        raise gr.Error(str(error))


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
        f"O que você gerou — {ours_count} arquivos, {_human(ours_size)}<br>"
        f"Cache de uploads — {gradio_count} arquivos, {_human(gradio_size)}"
    )


def clear_cache(include_gradio):
    """Apaga as saídas geradas. Nunca toca nos pesos dos modelos em ~/.cache/torch,
    porque apagá-los só forçaria baixar tudo de novo."""
    freed = 0
    for folder in [OUTPUT_DIR] + ([GRADIO_CACHE] if include_gradio else []):
        if not folder.exists():
            continue
        for item in folder.rglob("*"):
            if item.is_file():
                try:
                    freed += item.stat().st_size
                    item.unlink()
                except OSError:
                    pass

    gr.Info(f"{_human(freed)} liberados.")
    return cache_usage()


# ---------------------------------------------------------------- interface

with gr.Blocks(title="Dream Canvas") as demo:
    with gr.Column(elem_id="dc-header"):
        gr.HTML(
            "<h1>Dream Canvas</h1>"
            f"<p>O DeepDream original do Google, de 2015, rodando na sua máquina "
            f"em <code>{'ZeroGPU' if ZEROGPU else _startup_device.type}</code>.</p>"
        )

    with gr.Tab("Imagem"):
        with gr.Row():
            with gr.Column(scale=2, elem_classes="dc-rail"):
                image = gr.Image(type="pil", label="Sua imagem", height=260)
                preset = gr.Radio(list(PRESETS), value="Clássico 2015",
                                  label="Estilo", elem_classes="dc-preset")
                run_button = gr.Button("Sonhar", variant="primary", size="lg")

                with gr.Accordion("Ajuste fino", open=False):
                    model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Modelo")
                    layers = gr.CheckboxGroup(LAYER_CHOICES, value=["inception_4c/output"],
                                              label="Camadas")
                    iterations = gr.Slider(1, 100, value=DEFAULT_ITERATIONS, step=1,
                                           label="Iterações",
                                           info="Mais formas, mais tempo.")
                    step_size = gr.Slider(0.1, 6.0, value=DEFAULT_STEP_SIZE, step=0.1,
                                          label="Força",
                                          info="Quanto cada iteração empurra a imagem.")
                    octaves = gr.Slider(1, 8, value=DEFAULT_OCTAVES, step=1,
                                        label="Escalas",
                                        info="Em quantos tamanhos o efeito é aplicado.")
                    octave_scale = gr.Slider(1.1, 2.0, value=DEFAULT_OCTAVE_SCALE, step=0.05,
                                             label="Salto entre escalas")
                    jitter = gr.Slider(0, 64, value=DEFAULT_JITTER, step=1,
                                       label="Tremor",
                                       info="Desloca a imagem a cada passo. Evita emendas.")
                    max_dim = gr.Slider(256, MAX_DIM_CAP, value=min(1024, MAX_DIM_CAP),
                                        step=128, label="Resolução")
                    mode = gr.Radio(MODE_CHOICES, value="l2", label="Modo",
                                    info="Clássico é a receita de 2015.")
                    seed = gr.Number(value=None, label="Semente", precision=0,
                                     info="Repete o mesmo resultado. Exato só em CPU.")

            with gr.Column(scale=3):
                with gr.Column(elem_classes="dc-stage"):
                    comparison = gr.ImageSlider(label="Antes e depois", height=520)
                download = gr.File(label="Baixar PNG")

        with gr.Accordion("O que cada camada faz", open=False):
            gr.Markdown(LAYER_GUIDE, elem_classes="dc-reading")

        if os.path.exists("exemplo.jpg"):
            gr.Examples([["exemplo.jpg"]], inputs=[image], label="Experimente com esta")

    if VIDEO_OK:
        with gr.Tab("Vídeo"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes="dc-rail"):
                    video_in = gr.Video(label="Seu vídeo", height=260)
                    video_preset = gr.Radio(list(PRESETS), value="Clássico 2015",
                                            label="Estilo", elem_classes="dc-preset")
                    with gr.Row():
                        video_start = gr.Number(value=0, label="Começa em (s)", precision=1)
                        video_duration = gr.Number(value=3, label="Dura (s)", precision=1)
                    gr.Markdown("Cerca de 0,7 s de processamento por quadro. "
                                "Três segundos de vídeo levam perto de um minuto.",
                                elem_classes="dc-hint")
                    video_button = gr.Button("Sonhar", variant="primary", size="lg")

                    with gr.Accordion("Ajuste fino", open=False):
                        video_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL,
                                                  label="Modelo")
                        video_layers = gr.CheckboxGroup(LAYER_CHOICES,
                                                        value=["inception_4c/output"],
                                                        label="Camadas")
                        video_iterations = gr.Slider(
                            1, 40, value=video_module.DEFAULT_ITERATIONS, step=1,
                            label="Iterações por quadro")
                        video_step = gr.Slider(0.1, 6.0, value=DEFAULT_STEP_SIZE, step=0.1,
                                               label="Força")
                        video_octaves = gr.Slider(
                            1, 6, value=video_module.DEFAULT_OCTAVES, step=1, label="Escalas")
                        video_blend = gr.Slider(
                            0.0, 0.95, value=video_module.DEFAULT_BLEND, step=0.05,
                            label="Memória entre quadros",
                            info="Mais alto: mais estável, mais rastro.")
                        video_max_dim = gr.Slider(256, 1280,
                                                  value=video_module.DEFAULT_MAX_DIM,
                                                  step=64, label="Resolução")
                        video_flow = gr.Checkbox(
                            value=True, label="Acompanhar o movimento",
                            info="Desligar acelera, mas o padrão descola da cena.")

                with gr.Column(scale=3, elem_classes="dc-stage"):
                    video_out = gr.Video(label="Resultado", height=520)

        with gr.Tab("Zoom"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes="dc-rail"):
                    zoom_image = gr.Image(type="pil", label="Sua imagem", height=240)
                    zoom_center = gr.State((0.5, 0.5))
                    zoom_center_label = gr.Markdown(
                        "Clique na imagem para escolher o destino do zoom.",
                        elem_classes="dc-hint")
                    zoom_preset = gr.Radio(list(PRESETS), value="Clássico 2015",
                                           label="Estilo", elem_classes="dc-preset")
                    zoom_duration = gr.Slider(1, 30, value=5, step=1, label="Duração (s)")
                    zoom_speed = gr.Slider(
                        0.005, 0.08, value=video_module.DEFAULT_ZOOM, step=0.005,
                        label="Velocidade",
                        info="Lento hipnotiza; rápido não dá tempo das criaturas se formarem.")
                    gr.Markdown("Cerca de 0,4 s por quadro. "
                                "Cinco segundos levam perto de 40 s.",
                                elem_classes="dc-hint")
                    zoom_button = gr.Button("Sonhar", variant="primary", size="lg")

                    with gr.Accordion("Ajuste fino", open=False):
                        zoom_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL,
                                                 label="Modelo")
                        zoom_layers = gr.CheckboxGroup(LAYER_CHOICES,
                                                       value=["inception_4c/output"],
                                                       label="Camadas")
                        zoom_iterations = gr.Slider(
                            1, 40, value=video_module.DEFAULT_ITERATIONS, step=1,
                            label="Iterações por quadro")
                        zoom_step = gr.Slider(0.1, 6.0, value=DEFAULT_STEP_SIZE, step=0.1,
                                              label="Força")
                        zoom_octaves = gr.Slider(
                            1, 6, value=video_module.DEFAULT_OCTAVES, step=1, label="Escalas")
                        zoom_fps = gr.Slider(10, 30, value=video_module.DEFAULT_ZOOM_FPS,
                                             step=1, label="Quadros por segundo")
                        zoom_max_dim = gr.Slider(256, 1024, value=512, step=64,
                                                 label="Resolução")

                with gr.Column(scale=3):
                    with gr.Column(elem_classes="dc-stage"):
                        zoom_out = gr.Video(label="Resultado", height=380)
                    zoom_marked = gr.Image(label="Onde o zoom vai chegar", height=200,
                                           interactive=False)

    with gr.Accordion("Arquivos e espaço", open=False):
        usage = gr.Markdown(cache_usage(), elem_classes="dc-hint")
        gr.Markdown(
            f"Tudo que você gera fica em `{OUTPUT_DIR}` e some depois de "
            f"{int(OUTPUT_MAX_AGE / 3600)} h. Baixe o que quiser guardar. "
            f"Os modelos baixados nunca são apagados — removê-los só faria "
            f"você esperar o download de novo.",
            elem_classes="dc-hint")
        include_gradio = gr.Checkbox(
            value=False, label="Apagar também o cache de uploads",
            info="Resultados já na tela podem parar de carregar.")
        with gr.Row():
            refresh_button = gr.Button("Recontar", size="sm")
            clear_button = gr.Button("Apagar", variant="stop", size="sm")

    # ------------------------------------------------------------ ligações
    preset.change(apply_preset, [preset],
                  [model, layers, iterations, step_size, octaves, octave_scale])
    run_button.click(run,
                     [image, model, layers, iterations, step_size, octaves,
                      octave_scale, jitter, max_dim, mode, seed],
                     [comparison, download])

    if VIDEO_OK:
        video_preset.change(apply_motion_preset, [video_preset],
                            [video_model, video_layers, video_step])
        video_button.click(run_video,
                           [video_in, video_model, video_layers, video_iterations,
                            video_step, video_octaves, video_blend, video_max_dim,
                            video_flow, video_start, video_duration],
                           video_out)

        zoom_preset.change(apply_motion_preset, [zoom_preset],
                           [zoom_model, zoom_layers, zoom_step])
        zoom_image.select(mark_center, [zoom_image],
                          [zoom_marked, zoom_center, zoom_center_label])
        zoom_button.click(run_zoom,
                          [zoom_image, zoom_center, zoom_model, zoom_layers,
                           zoom_iterations, zoom_step, zoom_octaves, zoom_duration,
                           zoom_fps, zoom_speed, zoom_max_dim],
                          zoom_out)

    refresh_button.click(cache_usage, outputs=usage)
    clear_button.click(clear_cache, [include_gradio], usage)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=dc_theme.build_theme(),
        css=dc_theme.CSS,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("DEEPDREAM_SHARE") == "1",
    )
