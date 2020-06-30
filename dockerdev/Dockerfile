FROM python:3.6-slim 
RUN apt update && \
    apt install -y git build-essential wget 
ADD artyins-classifierservice /artyins-classifierservice
RUN cd artyins-classifierservice && \
    pip install -r requirements.txt 
WORKDIR artyins-classifierservice
ENTRYPOINT ["gunicorn","--bind","0.0.0.0:9891","--workers","10","wsgi:app"]

