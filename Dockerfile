FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8099 PUID=1000 PGID=1000
WORKDIR /app
COPY pyproject.toml README.md ./
COPY climate_optimizer ./climate_optimizer
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN pip install --no-cache-dir . && chmod 755 /usr/local/bin/entrypoint && mkdir -p /data
VOLUME /data
EXPOSE 8099
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
# the entrypoint hands /data to PUID:PGID and runs the service as that user
ENTRYPOINT ["entrypoint"]
CMD ["python", "-m", "climate_optimizer"]
