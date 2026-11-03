FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -r pepar && mkdir -p cache && chown pepar cache
USER pepar
# 설정 파일과 번역 내역 위치만 여기서 (나머지는 config.toml)
ENV PEPAR_CACHE_DIR=/app/cache PEPAR_CONFIG=/app/conf/config.toml PYTHONPATH=/app/pepar
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; from config import settings as s; h = s['host'][0]; h = '127.0.0.1' if h in ('0.0.0.0', '::') else h; urllib.request.urlopen(f'http://{h}:{s[\"port\"]}/login', timeout=3)"
CMD ["python", "pepar/app.py"]
