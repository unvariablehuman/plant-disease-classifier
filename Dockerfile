FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

Dan update `requirements.txt` jadi:
```
streamlit==1.32.0
tensorflow-cpu==2.15.0
numpy
Pillow
gdown
