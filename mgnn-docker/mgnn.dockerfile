FROM python:3.11

WORKDIR /mgnn-docker

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "mgnn-flask.py"]
