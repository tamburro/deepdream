FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache/torch \
    DEEPDREAM_DEVICE=cpu \
    DEEPDREAM_MAX_DIM=1024 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# A roda de CPU do torch é bem menor que a padrão, que traz CUDA junto.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY deepdream.py app.py exemplo.jpg ./

# Baixa os pesos na build, para o primeiro pedido não esperar o download.
RUN python -c "from deepdream import MODELS; MODELS['bvlc'][0]()"

EXPOSE 7860
CMD ["python", "app.py"]
