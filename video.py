"""DeepDream em vídeo, com coerência temporal.

Sonhar cada quadro de forma independente produz um flicker violento: o processo
é caótico e dois quadros quase iguais divergem completamente. Aqui o quadro
sonhado anterior é deformado pelo fluxo óptico até a posição do quadro atual e
misturado a ele antes de sonhar, o que faz os padrões grudarem nos objetos e
acompanharem o movimento.

Decodifica e codifica via pipes do ffmpeg, sem despejar milhares de PNGs em disco.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from deepdream import (
    DEFAULT_JITTER,
    DEFAULT_LAYER,
    DEFAULT_MODEL,
    DEFAULT_OCTAVE_SCALE,
    DEFAULT_STEP_SIZE,
    FeatureExtractor,
    dream,
    guide_features,
    pick_device,
)


def _prepare_guide(guide, model, layers, device):
    """Ativações da guia, calculadas uma vez para o vídeo inteiro."""
    if guide is None:
        return None
    extractor = FeatureExtractor(model, layers, device)
    return guide_features(extractor, guide, device)

# Padrões pensados para vídeo, não para imagem isolada: menos iterações por
# quadro, porque o efeito se acumula ao longo dos quadros pela realimentação.
DEFAULT_ITERATIONS = 6
DEFAULT_OCTAVES = 3
DEFAULT_BLEND = 0.6
DEFAULT_MAX_DIM = 640


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)

    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])

    num, den = video["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": float(info["format"].get("duration", 0)),
        "has_audio": has_audio,
    }


def audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def audio_envelope(path, fps, frames, rate=22050, smooth=3):
    """Energia do áudio por quadro, normalizada em [0, 1].

    Decodifica via ffmpeg para PCM mono e mede o RMS da janela de cada quadro.
    Normaliza pelo percentil 95 em vez do máximo, para um único pico não
    achatar o resto da música.
    """
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True, check=True,
    ).stdout
    samples = np.frombuffer(raw, np.float32)

    window = max(1, int(rate / fps))
    energy = np.array([
        float(np.sqrt(np.mean(np.square(samples[i * window : (i + 1) * window]))))
        if (i + 1) * window <= len(samples) else 0.0
        for i in range(frames)
    ])

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        energy = np.convolve(energy, kernel, mode="same")

    ceiling = np.percentile(energy, 95) or 1.0
    return np.clip(energy / ceiling, 0.0, 1.0)


def output_size(width, height, max_dim):
    scale = min(max_dim / max(width, height), 1.0)
    # O libx264 exige dimensões pares.
    return (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))


def decoder(path, size, start, duration):
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"scale={size[0]}:{size[1]}", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def encoder(path, size, fps, source, has_audio, start, duration):
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
    ]
    if has_audio:
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(source)]
        if duration:
            cmd += ["-t", str(duration)]
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _write_frame(write, data):
    """Escreve um quadro. Devolve False se o ffmpeg já fechou a entrada.

    Com `-shortest` o ffmpeg encerra assim que a trilha mais curta acaba, e os
    últimos quadros encontram o cano fechado. Isso é fim normal, não erro.
    """
    try:
        write.stdin.write(data)
        return True
    except BrokenPipeError:
        return False


def warp(image, previous_gray, current_gray):
    """Leva `image` (alinhada ao quadro anterior) para a posição do quadro atual.

    O fluxo é calculado do quadro atual para o anterior porque o remap busca,
    para cada pixel de destino, onde ele estava na origem.
    """
    flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


DEFAULT_ZOOM = 0.02
DEFAULT_ZOOM_FPS = 20


def zoom_step(array, center, amount):
    """Aproxima `amount` em direção a `center`, mantendo esse ponto fixo.

    center é normalizado (0 a 1). A matriz afim escala por (1 + amount) e
    translada de modo que o ponto escolhido caia sobre si mesmo.
    """
    height, width = array.shape[:2]
    scale = 1.0 + amount
    cx, cy = center[0] * width, center[1] * height
    matrix = np.float32([
        [scale, 0, cx * (1 - scale)],
        [0, scale, cy * (1 - scale)],
    ])
    return cv2.warpAffine(
        array, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def zoom_video(
    image,
    output,
    center=(0.5, 0.5),
    duration=None,
    fps=DEFAULT_ZOOM_FPS,
    zoom=DEFAULT_ZOOM,
    layers=(DEFAULT_LAYER,),
    model=DEFAULT_MODEL,
    iterations=DEFAULT_ITERATIONS,
    step_size=DEFAULT_STEP_SIZE,
    octaves=DEFAULT_OCTAVES,
    octave_scale=DEFAULT_OCTAVE_SCALE,
    jitter=DEFAULT_JITTER,
    objective="l2",
    guide=None,
    text=None,
    seed=0,
    device=None,
    max_dim=DEFAULT_MAX_DIM,
    audio=None,
    reactivity=1.0,
    on_frame=None,
    verbose=False,
):
    """Zoom infinito a partir de uma imagem. Devolve o caminho de saída.

    Com `audio`, a velocidade do zoom e a força do passo pulsam junto com a
    energia da faixa, e o áudio entra no vídeo final. Sem `duration`, a duração
    passa a ser a da própria faixa.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado. Instale com: brew install ffmpeg")

    device = pick_device(device)
    guides = None if text else _prepare_guide(guide, model, layers, device)

    image = image.convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)

    frame = np.array(image)
    # O libx264 exige dimensões pares.
    height, width = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
    frame = frame[:height, :width]

    # Com áudio e sem duração explícita, a faixa manda.
    if not duration:
        duration = audio_duration(audio) if audio is not None else 5.0

    total = max(1, int(duration * fps))
    envelope = audio_envelope(audio, fps, total) if audio is not None else None

    # O encoder já sabe puxar áudio de um arquivo de origem.
    write = encoder(output, (width, height), fps, audio, audio is not None, None, None)

    if verbose:
        print(f"{width}x{height} @ {fps}fps, {total} quadros, "
              f"centro em ({center[0]:.2f}, {center[1]:.2f})")

    try:
        for index in range(total):
            # A energia do quadro modula tanto o avanço quanto a intensidade.
            pulse = float(envelope[index]) if envelope is not None else 0.0
            zoom_now = zoom * (1.0 + reactivity * pulse)
            step_now = step_size * (1.0 + 0.5 * reactivity * pulse)

            dreamed = np.array(dream(
                Image.fromarray(frame),
                layers=layers, model=model, iterations=iterations,
                step_size=step_now, octaves=octaves, octave_scale=octave_scale,
                jitter=jitter, max_dim=None, objective=objective, guide=guides, text=text,
                seed=seed, device=device,
            ))
            if not _write_frame(write, dreamed.tobytes()):
                break

            # O próximo quadro parte do atual já aproximado: é isso que faz o
            # zoom parecer infinito, com detalhe novo nascendo no centro.
            frame = zoom_step(dreamed, center, zoom_now)

            if on_frame:
                on_frame(index + 1, total)
            if verbose:
                print(f"\rQuadro {index + 1}/{total}", end="", flush=True)
    finally:
        if verbose:
            print()
        if not write.stdin.closed:
            write.stdin.close()
        write.wait()

    return output


def process(
    source,
    output,
    layers=(DEFAULT_LAYER,),
    model=DEFAULT_MODEL,
    iterations=DEFAULT_ITERATIONS,
    step_size=DEFAULT_STEP_SIZE,
    octaves=DEFAULT_OCTAVES,
    octave_scale=DEFAULT_OCTAVE_SCALE,
    jitter=DEFAULT_JITTER,
    objective="l2",
    guide=None,
    text=None,
    seed=0,
    device=None,
    max_dim=DEFAULT_MAX_DIM,
    fps=None,
    start=None,
    duration=None,
    blend=DEFAULT_BLEND,
    flow=True,
    audio=True,
    on_frame=None,
    verbose=False,
):
    """Aplica DeepDream a um vídeo inteiro. Devolve o caminho de saída."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado. Instale com: brew install ffmpeg")

    device = pick_device(device)
    guides = None if text else _prepare_guide(guide, model, layers, device)

    info = probe(source)
    size = output_size(info["width"], info["height"], max_dim)
    fps = fps or info["fps"]
    frame_bytes = size[0] * size[1] * 3

    span = duration or max(0.0, info["duration"] - (start or 0))
    expected = int(span * info["fps"]) if span else 0

    if verbose:
        print(f"{info['width']}x{info['height']} @ {info['fps']:.2f}fps "
              f"-> {size[0]}x{size[1]} @ {fps:.2f}fps"
              + (f", ~{expected} quadros" if expected else ""))

    read = decoder(source, size, start, duration)
    write = encoder(output, size, fps, source, info["has_audio"] and audio, start, duration)

    previous_dream = None
    previous_gray = None
    index = 0

    try:
        while True:
            raw = read.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            base = frame.astype(np.float32)

            if previous_dream is not None:
                carried = previous_dream.astype(np.float32)
                if flow:
                    carried = warp(carried, previous_gray, gray)
                base = (1 - blend) * base + blend * carried

            dreamed = dream(
                Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)),
                layers=layers,
                model=model,
                iterations=iterations,
                step_size=step_size,
                octaves=octaves,
                octave_scale=octave_scale,
                jitter=jitter,
                max_dim=None,
                objective=objective,
                # A mesma seed em todo quadro mantém o padrão de jitter idêntico,
                # o que reduz bastante o flicker residual.
                seed=seed,
                device=device,
            )

            previous_dream = np.array(dreamed)
            previous_gray = gray
            if not _write_frame(write, previous_dream.tobytes()):
                break

            index += 1
            if on_frame:
                on_frame(index, expected)
            if verbose:
                suffix = f"/{expected}" if expected else ""
                print(f"\rQuadro {index}{suffix}", end="", flush=True)
    finally:
        if verbose:
            print()
        read.stdout.close()
        read.wait()
        if not write.stdin.closed:
            write.stdin.close()
        write.wait()

    if index == 0:
        raise RuntimeError("Nenhum quadro foi lido. O arquivo de entrada é um vídeo válido?")
    if verbose:
        print(f"Salvo em {output} ({index} quadros)")
    return output


def main():
    parser = argparse.ArgumentParser(description="DeepDream em vídeo.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-l", "--layers", default=DEFAULT_LAYER)
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-n", "--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--octaves", type=int, default=DEFAULT_OCTAVES)
    parser.add_argument("--octave-scale", type=float, default=DEFAULT_OCTAVE_SCALE)
    parser.add_argument("--jitter", type=int, default=DEFAULT_JITTER)
    parser.add_argument("--objective", choices=["l2", "mean"], default="l2")
    parser.add_argument("-g", "--guide", type=Path,
                        help="Imagem-guia: os quadros puxam para as formas dela")
    parser.add_argument("-t", "--text",
                        help="Descrição para o CLIP perseguir. Bem mais lento")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--start", type=float, help="Segundo inicial do trecho")
    parser.add_argument("--duration", type=float, help="Duração do trecho, em segundos")
    parser.add_argument(
        "--blend", type=float, default=DEFAULT_BLEND,
        help="Quanto do quadro sonhado anterior é realimentado (0 a 1). "
             "Mais alto: mais estável e com mais rastro.",
    )
    parser.add_argument("--audio", type=Path,
                        help="Faixa que dirige o pulso do zoom (só no modo zoom)")
    parser.add_argument("--reactivity", type=float, default=1.0,
                        help="Quanto o som afeta zoom e força. 0 desliga")
    parser.add_argument("--no-flow", action="store_true",
                        help="Não usar fluxo óptico (mais rápido, menos estável)")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_dream.mp4")

    try:
        process(
            args.input, output,
            layers=[s for s in args.layers.split(",") if s.strip()],
            model=args.model,
            iterations=args.iterations,
            step_size=args.step_size,
            octaves=args.octaves,
            octave_scale=args.octave_scale,
            jitter=args.jitter,
            objective=args.objective,
            guide=Image.open(args.guide) if args.guide else None,
            text=args.text,
            seed=args.seed,
            device=args.device,
            max_dim=args.max_dim,
            fps=args.fps,
            start=args.start,
            duration=args.duration,
            blend=args.blend,
            flow=not args.no_flow,
            audio=not args.no_audio,
            verbose=True,
        )
    except RuntimeError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
