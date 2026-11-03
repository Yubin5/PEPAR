FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -r pepar && mkdir -p cache && chown pepar cache
USER pepar
# 컨테이너 안에서 정해진 값만 여기서: 설정 파일 위치(compose.yml이 연결하는 폴더)와 번역 내역 위치(pepar-cache 볼륨).
# 나머지 설정은 config.toml (주소·포트도: 호스트 네트워크로 실행한다)
ENV PEPAR_CACHE_DIR=/app/cache PEPAR_CONFIG=/app/conf/config.toml PYTHONPATH=/app/pepar
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; from config import settings as s; h = s['host'][0]; h = '127.0.0.1' if h in ('0.0.0.0', '::') else h; urllib.request.urlopen(f'http://{h}:{s[\"port\"]}/login', timeout=3)"
CMD ["python", "pepar/app.py"]
