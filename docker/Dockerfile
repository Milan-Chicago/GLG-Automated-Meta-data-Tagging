FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app/main.py /app/app/main.py
COPY ./pretrained_models /app/pretrained_models
COPY requirements.txt /app/requirements.txt
RUN pip --no-cache-dir install -r requirements.txt
RUN apt-get update
