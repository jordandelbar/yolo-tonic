FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender1 ffmpeg

CMD ["python", "main.py"]
