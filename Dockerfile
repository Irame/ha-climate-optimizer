FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8099
WORKDIR /app
COPY pyproject.toml README.md ./
COPY climate_optimizer ./climate_optimizer
RUN pip install --no-cache-dir . && useradd --system --uid 1000 optimizer && mkdir -p /data && chown optimizer /data
USER optimizer
VOLUME /data
EXPOSE 8099
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["python", "-m", "climate_optimizer"]
