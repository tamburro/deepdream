FROM python:3.12-slim

# O pacote pytorch-caffe-models é instalado direto do GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# O Hugging Face Spaces roda o container como o usuário 1000.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/user/.cache/torch \
    DEEPDREAM_DEVICE=cpu \
    DEEPDREAM_MAX_DIM=1024 \
    PORT=7860

WORKDIR $HOME/app

# As wheels de CPU do torch são ~10x menores que as padrão, que trazem CUDA junto.
RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user deepdream.py app.py exemplo.jpg ./

# Baixa os pesos na build, para o primeiro visitante não esperar o download.
RUN python -c "from deepdream import MODELS; MODELS['bvlc'][0]()"

EXPOSE 7860
CMD ["python", "app.py"]
