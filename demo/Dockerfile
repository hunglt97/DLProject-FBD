FROM python:3.8
WORKDIR /app
COPY fer_web/requirements.txt .
RUN pip install -r requirements.txt
COPY fer_web/ .
ENV PATH=/root/.local:$PATH
EXPOSE 80
CMD ["python", "./server_fer.py"]