FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY src/ src/
COPY configuration/ configuration/
RUN pip install -r requirements.txt --no-cache-dir

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender1 ffmpeg

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
