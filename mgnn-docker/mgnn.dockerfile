FROM python:3.9

WORKDIR /mgnn-docker

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "mgnn-flask.py"]
