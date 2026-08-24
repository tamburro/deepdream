"""Baixa conjuntos de imagens do Wikimedia Commons para treinar.

Por que a Commons e não uma raspagem de busca de imagens: aqui a licença de
cada arquivo vem declarada, a árvore de categorias já dá os **rótulos** que o
fine-tuning precisa, e a API devolve a imagem redimensionada no servidor, na
largura pedida — não é preciso baixar originais de 20 MB para treinar a 224px.

    .venv/bin/python dataset.py --category "Botanical illustrations" --depth 2

Cada subcategoria vira uma pasta, que o `ImageFolder` do torchvision lê como
classe. Um `manifest.csv` guarda origem, licença e autoria de cada arquivo —
sem isso não há como publicar nem monetizar o que sair do treino.
"""

import argparse
import collections
import csv
import hashlib
import io
import shutil
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"

# A Commons pede um User-Agent que identifique o cliente e um contato.
USER_AGENT = (
    "DreamCanvas-dataset/1.0 "
    "(https://github.com/tamburro/deepdream; treino de fine-tuning)"
)

# Licenças aceitas por padrão. Tudo que não bate é descartado, porque um
# conjunto de treino com licença desconhecida contamina o que for gerado.
OPEN_LICENSES = ("public domain", "cc0", "cc by", "cc-by", "pd-")

# CC BY-SA é copyleft: o compartilhamento nas mesmas condições pode alcançar o
# que se deriva dele. Para treinar um modelo que talvez seja monetizado, o modo
# estrito fica só no que é inequivocamente livre.
STRICT_LICENSES = ("public domain", "cc0", "pd-")


def _get(params, retries=3):
    params = {**params, "format": "json"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def subcategories(category, depth=1, _seen=None):
    """Subcategorias, descendo `depth` níveis. Cada uma vira uma classe."""
    if depth < 1:
        return []
    _seen = _seen if _seen is not None else set()

    found = []
    cont = {}
    while True:
        data = _get({
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": "subcat",
            "cmlimit": "500", **cont,
        })
        for member in data.get("query", {}).get("categorymembers", []):
            name = member["title"].removeprefix("Category:")
            if name in _seen:
                continue
            _seen.add(name)
            found.append(name)
            found.extend(subcategories(name, depth - 1, _seen))
        cont = data.get("continue", {})
        if not cont:
            break
    return found


def files_in(category, limit, width, depth=2, _seen=None):
    """Arquivos de uma categoria, já com a URL da miniatura na largura pedida.

    Muitas categorias da Commons guardam só subcategorias, sem arquivo direto.
    Por isso, quando a categoria não enche a cota, descemos atrás dos arquivos
    — senão metade das classes volta vazia.
    """
    _seen = _seen if _seen is not None else set()
    if category in _seen:
        return []
    _seen.add(category)

    out = []
    cont = {}
    while len(out) < limit:
        data = _get({
            "action": "query", "generator": "categorymembers",
            "gcmtitle": f"Category:{category}", "gcmtype": "file",
            "gcmlimit": "200", "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "iiurlwidth": str(width), **cont,
        })
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("thumburl"):
                continue
            meta = info.get("extmetadata", {})
            out.append({
                "title": page["title"].removeprefix("File:"),
                "url": info["thumburl"],
                "source": info.get("descriptionurl", ""),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "artist": _strip_html(meta.get("Artist", {}).get("value", "")),
            })
            if len(out) >= limit:
                break
        cont = data.get("continue", {})
        if not cont:
            break

    if len(out) < limit and depth > 0:
        for child in subcategories(category, 1, set(_seen)):
            if len(out) >= limit:
                break
            out.extend(files_in(child, limit - len(out), width, depth - 1, _seen))

    return out


def _strip_html(text):
    out, tag = [], False
    for ch in text:
        if ch == "<":
            tag = True
        elif ch == ">":
            tag = False
        elif not tag:
            out.append(ch)
    return " ".join("".join(out).split())[:120]


def is_open(license_text, strict=False):
    lowered = license_text.lower()
    allowed = STRICT_LICENSES if strict else OPEN_LICENSES
    return any(token in lowered for token in allowed)


FAILURES = collections.Counter()


def fetch(item, folder, min_side, retries=4):
    """Baixa, valida e grava em JPEG. Devolve o nome do arquivo ou None."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in item["title"])
    # Sufixo com hash do título: sem ele, títulos que só diferem em pontuação
    # colapsam no mesmo nome e um sobrescreve o outro, em silêncio.
    digest = hashlib.sha1(item["title"].encode()).hexdigest()[:8]
    name = f"{safe[:80]}-{digest}.jpg"
    target = folder / name
    if target.exists():
        return name

    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                item["url"], headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            break
        except urllib.error.HTTPError as error:
            if attempt == retries - 1:
                FAILURES[f"HTTP {error.code}"] += 1
                return None
            # A Wikimedia diz quanto esperar quando limita a taxa; ignorar isso
            # e tentar de novo na hora só faz o próximo pedido falhar também.
            wait = error.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else 2.0 * (attempt + 1))
        except Exception as error:
            if attempt == retries - 1:
                FAILURES[type(error).__name__] += 1
                return None
            time.sleep(2.0 * (attempt + 1))

    # Abaixo disso o recorte de 224 do treino viraria ampliação de imagem pobre.
    if min(image.size) < min_side:
        FAILURES["pequena"] += 1
        return None

    image.save(target, "JPEG", quality=92)
    return name


def build(category, out_dir, depth, per_class, width, min_side, workers,
          only_open, min_per_class=20, strict=False, max_classes=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = subcategories(category, depth) or [category]
    if max_classes:
        classes = classes[:max_classes]
    print(f"{len(classes)} classes a partir de '{category}' "
          f"(profundidade {depth})", flush=True)

    manifest = out_dir / "manifest.csv"
    is_new = not manifest.exists()
    total = 0

    with open(manifest, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["classe", "arquivo", "titulo", "licenca", "autoria", "origem"])

        for index, name in enumerate(classes, 1):
            done = out_dir / "".join(
                c if c.isalnum() or c in "-_" else "_" for c in name
            )[:80]
            if done.is_dir() and len(list(done.glob("*.jpg"))) >= per_class:
                print(f"  [{index}/{len(classes)}] {name[:44]:46} "
                      f"— já completa, pulando", flush=True)
                continue
            try:
                items = files_in(name, per_class, width)
            except Exception as error:
                print(f"  [{index}/{len(classes)}] {name[:44]:46} "
                      f"— falhou ({type(error).__name__}), seguindo", flush=True)
                continue
            if only_open:
                items = [i for i in items if is_open(i["license"], strict)]
            if len(items) < min_per_class:
                print(f"  [{index}/{len(classes)}] {name[:44]:46} "
                      f"— só {len(items)} candidatos, abaixo do mínimo", flush=True)
                continue

            folder = out_dir / "".join(
                c if c.isalnum() or c in "-_" else "_" for c in name
            )[:80]
            folder.mkdir(exist_ok=True)

            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(
                        pool.map(lambda i: (i, fetch(i, folder, min_side)), items)
                    )
            except Exception as error:
                print(f"  [{index}/{len(classes)}] {name[:44]:46} "
                      f"— erro ao baixar ({type(error).__name__}), seguindo", flush=True)
                continue

            kept = [(i, f) for i, f in results if f]

            # O mínimo tem de valer sobre o que foi **salvo**: muitos candidatos
            # caem no filtro de resolução, e uma pasta com 3 arquivos — ou vazia
            # — quebra o ImageFolder e não serve para treinar.
            if len(kept) < min_per_class:
                # rmtree, não rmdir: a pasta pode ter sobras de outra execução,
                # e o rmdir estourava OSError e derrubava o processo inteiro.
                shutil.rmtree(folder, ignore_errors=True)
                print(f"  [{index}/{len(classes)}] {name[:44]:46} "
                      f"— só {len(kept)} baixadas, classe descartada", flush=True)
                continue

            for item, filename in kept:
                writer.writerow([folder.name, filename, item["title"],
                                 item["license"], item["artist"], item["source"]])
            handle.flush()
            total += len(kept)
            print(f"  [{index}/{len(classes)}] {name[:44]:46} "
                  f"{len(kept):4d} imagens", flush=True)

    if FAILURES:
        print("\nDescartes:", dict(FAILURES.most_common()))
    print(f"\n{total} imagens em {out_dir}")
    print(f"Procedência em {manifest}")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Baixa imagens do Wikimedia Commons para treino."
    )
    parser.add_argument("--category", default="Botanical illustrations",
                        help="Categoria raiz. As subcategorias viram classes")
    parser.add_argument("-o", "--out", default="dataset/botanical")
    parser.add_argument("--depth", type=int, default=1,
                        help="Níveis de subcategoria a descer")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--min-per-class", type=int, default=20,
                        help="Descarta classes com menos que isso. Classe minúscula "
                             "atrapalha o treino")
    parser.add_argument("--width", type=int, default=512,
                        help="Largura pedida à API. 512 basta para treinar a 224")
    parser.add_argument("--min-side", type=int, default=256,
                        help="Descarta imagens com o menor lado abaixo disso")
    parser.add_argument("--workers", type=int, default=2,
                        help="Medido: 2 workers acertam 75%% e 4 só 12%%, porque a Wikimedia limita a taxa")
    parser.add_argument("--all-licenses", action="store_true",
                        help="Não filtrar por licença aberta. Não recomendado")
    parser.add_argument("--max-classes", type=int,
                        help="Limita quantas classes baixar. Útil para testar antes "
                             "de disparar um conjunto grande")
    parser.add_argument("--strict-licenses", action="store_true",
                        help="Só domínio público e CC0, sem CC BY-SA")
    parser.add_argument("--list-only", action="store_true",
                        help="Só mostra as classes que seriam baixadas")
    args = parser.parse_args()

    if args.list_only:
        for name in subcategories(args.category, args.depth) or [args.category]:
            print(name)
        return

    try:
        build(args.category, args.out, args.depth, args.per_class, args.width,
              args.min_side, args.workers, not args.all_licenses,
              args.min_per_class, args.strict_licenses, args.max_classes)
    except KeyboardInterrupt:
        sys.exit("\nInterrompido. O que já baixou continua válido.")


if __name__ == "__main__":
    main()
