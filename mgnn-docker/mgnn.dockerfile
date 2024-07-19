FROM python:3.9-slim

WORKDIR /mgnn-docker

COPY . /mgnn-docker

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV NAME MGNN

CMD ["python", "mgnn-flask.py"]
