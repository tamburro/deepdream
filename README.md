---
title: Dream Canvas
emoji: 🐕
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: DeepDream clássico do Google (2015) com os pesos bvlc_googlenet
---

# Dream Canvas

O design system está em [DESIGN.md](DESIGN.md) e a implementação em Gradio, em
[theme.py](theme.py). O DESIGN.md é a fonte: mexa nele antes de mexer nos tokens.


Reprodução do DeepDream original do Google (Mordvintsev, 2015) em PyTorch, com
os pesos `bvlc_googlenet` do Caffe — não o InceptionV3 do tutorial em TensorFlow.
É esse modelo que produz as "puppy slugs" clássicas.

## Rodar local

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Abre em <http://localhost:7860>. Em Apple Silicon usa a GPU via MPS automaticamente.

## App de desktop

A mesma interface numa janela nativa, sem navegador:

```bash
.venv/bin/pip install -r requirements-desktop.txt
.venv/bin/python local.py
```

No macOS dá para deixar o `Dream Canvas.command` na mesa e abrir com dois cliques.

Para gerar um link público temporário (72 h), tunelando para a sua máquina:

```bash
DEEPDREAM_SHARE=1 .venv/bin/python app.py
```

## Linha de comando

```bash
.venv/bin/python deepdream.py foto.jpg -l inception_4c/output -o saida.png
```

Modelos: `bvlc` (padrão), `places365`, `places205`, `torchvision`.
Camadas: `inception_3a` a `inception_5b`.

## Vídeo

Aplica o efeito a um vídeo inteiro, com coerência temporal:

```bash
.venv/bin/pip install -r requirements-video.txt
.venv/bin/python video.py entrada.mp4 -o saida.mp4
```

Exige o `ffmpeg` no sistema (`brew install ffmpeg`). O áudio do original é
preservado. A interface (`app.py` e `local.py`) ganha uma aba **Vídeo** quando
o opencv e o ffmpeg estão disponíveis — ela não aparece no Hugging Face Spaces,
porque um vídeo estoura o limite de duração da GPU do ZeroGPU.

Sonhar cada quadro isoladamente pisca muito: o processo é caótico e dois quadros
quase idênticos divergem por completo. Aqui o quadro sonhado anterior é deformado
pelo fluxo óptico até a posição do quadro atual e misturado a ele antes de sonhar,
o que faz os padrões grudarem nos objetos e acompanharem o movimento. A mesma
seed em todos os quadros mantém o padrão de jitter idêntico, o que remove parte
do flicker residual.

Medido num teste com zoom, o flicker com movimento compensado (menor é melhor):

| | Flicker |
| --- | --- |
| Vídeo de entrada (piso do ruído) | 1,86 |
| Quadro a quadro, sem realimentação | 6,69 |
| Realimentado, sem fluxo óptico | 8,32 |
| Realimentado + fluxo óptico | **4,58** |

Note que realimentar **sem** fluxo é pior que não realimentar: o padrão fica
parado enquanto a cena se move. É por isso que o fluxo óptico é o padrão.

Opções úteis: `--blend` (quanto do quadro anterior é realimentado; mais alto dá
mais estabilidade e mais rastro), `--start` e `--duration` para recortar um
trecho, `--max-dim` para a resolução, e `--no-flow` para ganhar velocidade
abrindo mão da estabilidade.

## Zoom infinito

Gera um vídeo a partir de **uma** imagem: sonha um quadro, aproxima um pouco em
direção a um ponto escolhido, e repete. Como cada quadro parte do anterior já
aproximado, detalhe novo nasce no centro sem parar.

Na interface (aba **Zoom infinito**) você clica na imagem para mirar o ponto e
escolhe a duração. Por código:

```python
from PIL import Image
import video

video.zoom_video(Image.open("foto.jpg"), "zoom.mp4",
                 center=(0.28, 0.22), duration=5, zoom=0.025)
```

`center` é normalizado (0 a 1) e `zoom` é o quanto a imagem avança por quadro.
Conte ~0,4 s por quadro a 512px: 5 s a 20 fps são 100 quadros, cerca de 40 s.

## Plugin do Figma

Exporta a camada selecionada, aplica o efeito e insere o resultado de volta no
documento. Veja [figma-plugin/README.md](figma-plugin/README.md).

Ele conversa com a API HTTP, que também serve para qualquer outra integração:

```bash
.venv/bin/python server.py
```

## Docker

```bash
docker build -t deepdream .
docker run -p 7860:7860 deepdream
```

O Dockerfile serve para auto-hospedagem, e usa wheels de CPU do torch. No Hugging
Face Spaces o app usa o SDK `gradio` com hardware ZeroGPU — ambos gratuitos; o SDK
`docker` e o CPU Basic são pagos. Serverless não serve em nenhum caso: as
dependências passam de 1 GB e o Gradio precisa de um servidor de longa duração.

No ZeroGPU a GPU só existe dentro da função decorada com `@spaces.GPU`, por isso o
device é resolvido a cada execução e os pesos ficam em cache na CPU, movidos por
cópia para a GPU a cada chamada.

## Variáveis de ambiente

| Variável | Padrão | Para quê |
| --- | --- | --- |
| `DEEPDREAM_DEVICE` | melhor disponível | `mps`, `cuda` ou `cpu` |
| `DEEPDREAM_MAX_DIM` | 1536 (768 em CPU) | Teto de resolução |
| `DEEPDREAM_SHARE` | — | `1` gera link público temporário |
| `PORT` | 7860 | Porta do servidor |
| `DEEPDREAM_OUTPUT_DIR` | `$TMPDIR/deepdream` | Onde as saídas são gravadas |
| `DEEPDREAM_OUTPUT_MAX_AGE_H` | 24 | Horas até uma saída ser apagada |

A interface tem uma seção **Manutenção** com o uso atual e um botão para limpar.

### Onde ficam os arquivos

As imagens e vídeos gerados pela interface vão para uma pasta única
(`$TMPDIR/deepdream` por padrão, mostrada no topo da própria interface) com nome
`AAAAMMDD-HHMMSS-xxxxxx`. Saídas com mais de 24 h são apagadas quando o app
sobe. **Baixe o que quiser guardar** — ou aponte `DEEPDREAM_OUTPUT_DIR` para
uma pasta permanente.

A CLI é diferente: ela escreve onde você mandar, e nada é apagado.

Além disso, o Gradio mantém cache próprio de uploads em `$TMPDIR/gradio`
(controlado por `GRADIO_TEMP_DIR`), e os pesos dos modelos ficam em
`~/.cache/torch`, que são permanentes de propósito — apagá-los só força novo
download.
