FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3-pip python3.11-venv build-essential curl git wget git-lfs libssl-dev libffi-dev libgomp1 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python && pip install --upgrade pip

WORKDIR /app

RUN pip install --no-cache-dir torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Pin transformers FIRST so Unsloth installs against a known-good version.
# transformers >= 4.50 reorganized model classes (BloomPreTrainedModel removed),
# which breaks Unsloth's internal patching. 4.45.2 is the last stable version
# that works with Unsloth + TRL + PEFT together.
ARG CACHE_BUST_V2=2026-04-26b
RUN pip install --no-cache-dir "transformers==4.45.2" "trl>=0.8.6,<0.14" "peft>=0.7.0,<0.14" "datasets>=2.14.0" "accelerate>=0.27.0" "bitsandbytes>=0.43.0"

RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

RUN pip install --no-cache-dir "huggingface_hub>=0.22.0" "openai>=1.0.0" "gymnasium>=0.29.0" "numpy>=1.24.0" "python-dotenv>=1.0.0" "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" "pydantic>=2.0.0" "websockets>=12.0" "requests>=2.31.0" "python-multipart>=0.0.9" "httpx>=0.27.0" "gradio>=4.0.0,<6.0.0" "openenv-core[cli]>=0.1.1"

COPY guardian/ ./guardian/
COPY server/ ./server/
COPY models.py training_space.py setup.py ./

RUN pip install --no-cache-dir -e . --no-deps && mkdir -p guardian/data guardian/checkpoints

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "training_space.py"]
