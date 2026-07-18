# CPU-only, relocatable runtime for Windows Docker Desktop and macOS Docker
# Desktop. linux/amd64 is selected in compose for Apple Silicon compatibility.
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build
COPY requirements-app.txt requirements-ie.txt requirements-ocr.txt requirements-layout.txt ./

RUN python -m venv /opt/venvs/app \
    && /opt/venvs/app/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venvs/app/bin/pip install --no-cache-dir -r requirements-app.txt

RUN python -m venv /opt/venvs/ocr \
    && /opt/venvs/ocr/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venvs/ocr/bin/pip install --no-cache-dir paddlepaddle==3.3.0 \
       -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && /opt/venvs/ocr/bin/pip install --no-cache-dir torch==2.8.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venvs/ocr/bin/pip install --no-cache-dir -r requirements-ocr.txt

RUN python -m venv /opt/venvs/layout \
    && /opt/venvs/layout/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venvs/layout/bin/pip install --no-cache-dir torch==2.8.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venvs/layout/bin/pip install --no-cache-dir -r requirements-layout.txt

# PaddleOCR's current dependency graph can install a non-headless OpenCV wheel
# even though the project pins the headless package. Provide its small runtime
# dependency without changing the verified Python package set.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
EXPOSE 7860
CMD ["/opt/venvs/app/bin/python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
