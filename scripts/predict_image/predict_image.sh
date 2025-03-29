curl -X POST http://localhost:8000/predict_image \
     -H "Content-Type: image/jpeg" \
     --data-binary @baseball.jpg \
     --output baseball_annotated.jpg
