FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -r pepar && mkdir -p cache && chown pepar cache
USER pepar
ENV PEPAR_HOST=0.0.0.0
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/login', timeout=3)"
CMD ["python", "app.py"]
