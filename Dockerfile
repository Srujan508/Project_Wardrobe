# CPU image for serving the app (training happens locally on the GPU).
#   docker build -t wardrobe .
#   docker run -p 127.0.0.1:8000:8000 wardrobe          # then open http://localhost:8000
# To serve on a LAN, also pass  -e WARDROBE_ALLOWED_HOSTS=<your-hostname>  (the app has no login).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pinned versions from the lock file; CUDA/GPU wheels swapped for CPU builds,
# dev-only tools (pytest, ipykernel, ...) left out.
COPY requirements.lock.txt .
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
 && grep -viE '^(torch|torchvision|onnxruntime-gpu|playwright|pytest-playwright|pytest-base-url|pyee|python-slugify|text-unidecode|pytest|iniconfig|pluggy|ipykernel|ipython|jupyter|debugpy|matplotlib-inline|comm|tornado|nest-asyncio|traitlets|prompt-toolkit|pygments|jedi|parso|pexpect|ptyprocess|stack-data|executing|asttokens|pure-eval|wcwidth|decorator|psutil|pyzmq)==' requirements.lock.txt > req.txt \
 && grep -iE '^onnxruntime-gpu==' requirements.lock.txt | sed 's/onnxruntime-gpu/onnxruntime/' >> req.txt \
 && pip install -r req.txt \
 && rm req.txt

COPY configs/ configs/
COPY src/ src/
COPY app/ app/
# Trained model + calibrator (the app falls back to raw FashionCLIP without them)
COPY models/ models/

# Run as an unprivileged user; only uploads/ and the database are writable.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/uploads /app/data \
 && chown -R app:app /app/uploads /app/data
USER app
ENV WARDROBE_DB=/app/data/wardrobe.db HF_HOME=/home/app/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
