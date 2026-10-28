#!/bin/sh
# Starts as root only long enough to hand the data directory to PUID:PGID
# (a bind mount Docker creates belongs to root), then runs as that user.
set -e
DATA_DIR="${DATA_DIR:-/data}"
if [ "$(id -u)" = "0" ]; then
  PUID="${PUID:-1000}"
  PGID="${PGID:-1000}"
  mkdir -p "$DATA_DIR"
  if [ "$(stat -c %u:%g "$DATA_DIR")" != "$PUID:$PGID" ]; then
    chown -R "$PUID:$PGID" "$DATA_DIR"
  fi
  HOME=/tmp exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups -- "$@"
fi
if [ ! -w "$DATA_DIR" ]; then
  echo "climate-optimizer: $DATA_DIR is not writable for uid $(id -u)." >&2
  echo "Run 'chown -R $(id -u) <host directory>' or start the container without 'user:'," >&2
  echo "then it hands the directory to PUID:PGID (default 1000:1000) itself." >&2
  exit 1
fi
exec "$@"
