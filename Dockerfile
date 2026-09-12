# Dockerfile for Scientific ML on Zero Budget
# Base: PyTorch with CUDA runtime for local NVIDIA GPU testing
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy repository source code
COPY . /workspace/

# Expose default Jupyter port
EXPOSE 8888

# Launch JupyterLab by default or run demonstration script
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
