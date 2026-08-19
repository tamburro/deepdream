---
title: DeepDream
emoji: 🐕
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
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

O mesmo Dockerfile roda no Hugging Face Spaces (SDK `docker`), que é onde o app
está hospedado. Serverless não serve: as dependências passam de 1 GB e o Gradio
precisa de um servidor de longa duração.

## Variáveis de ambiente

| Variável | Padrão | Para quê |
| --- | --- | --- |
| `DEEPDREAM_DEVICE` | melhor disponível | `mps`, `cuda` ou `cpu` |
| `DEEPDREAM_MAX_DIM` | 1536 | Teto de resolução; baixe ao hospedar em CPU |
| `DEEPDREAM_SHARE` | — | `1` gera link público temporário |
| `PORT` | 7860 | Porta do servidor |
