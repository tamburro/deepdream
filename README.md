---
title: DeepDream
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

# DeepDream clássico

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

No macOS dá para deixar o `DeepDream.command` na mesa e abrir com dois cliques.

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
