FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -r pepar && mkdir -p cache && chown pepar cache
USER pepar
# 컨테이너 안에서 정해진 값만 여기서. 나머지 설정은 compose.yml이 연결하는 폴더의 config.toml (없으면 기본값):
# 컨테이너 밖으로는 compose.yml의 ports가 열고, 번역 내역은 pepar-cache 볼륨에 남는다
ENV PEPAR_HOST=0.0.0.0 PEPAR_PORT=8765 PEPAR_CACHE_DIR=/app/cache PEPAR_CONFIG=/app/conf/config.toml
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; from config import settings as s; h = s['host'][0]; h = '127.0.0.1' if h in ('0.0.0.0', '::') else h; urllib.request.urlopen(f'http://{h}:{s[\"port\"]}/login', timeout=3)"
CMD ["python", "app.py"]
