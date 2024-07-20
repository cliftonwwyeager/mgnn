FROM python:3.9-slim

WORKDIR /mgnn-docker

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /mgnn-docker

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio

RUN pip install redis

RUN pip install numpy pandas scikit-learn

RUN pip install optuna

EXPOSE 5000

ENV NAME World

CMD ["python", "mgnn-flask.py"]
