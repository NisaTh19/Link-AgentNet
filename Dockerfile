FROM python:3.10-slim

WORKDIR /app

# System tools needed to build some Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (must install before PyG)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric + sparse backends (CPU wheels)
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-geometric==2.4.0 \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Remaining dependencies
RUN pip install --no-cache-dir \
    networkx \
    scikit-learn \
    scipy \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    optuna \
    streamlit \
    plotly

# Copy only what the app needs at runtime
COPY src/app.py src/model.py src/util.py ./src/
COPY out/checkpoints/ ./out/checkpoints/
COPY out/datasets_splits/ ./out/datasets_splits/
COPY out/optuna_db/ ./out/optuna_db/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
