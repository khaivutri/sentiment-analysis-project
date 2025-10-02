FROM python:3.12

WORKDIR /src

# Copy requirements first for better caching
COPY requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY server.py /src/server.py
COPY models/ /src/models/
COPY utils/ /src/utils/
COPY .env /src/.env

# Create model cache directory (models will be downloaded from Google Drive)
RUN mkdir -p /src/model_cache

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]