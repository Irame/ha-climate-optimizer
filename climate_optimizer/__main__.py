"""python -m climate_optimizer: run the service."""

import logging
import os
import sys
from pathlib import Path

import uvicorn

log = logging.getLogger("climate_optimizer")


def writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError as e:
        log.error("%s is not writable for uid %s: %s. Make it writable for that user "
                  "(in Docker: start without 'user:', or set PUID/PGID to the directory's owner).", path, os.getuid(), e)
        return False


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data = Path(os.environ.get("DATA_DIR", "/data"))
    if os.environ.get("DEMO"):
        from .demo import demo_app, demo_dir
        if not writable(demo_dir()):
            sys.exit(1)
        app = demo_app()
    else:
        from .app import create_app
        if not writable(data):
            sys.exit(1)
        app = create_app()
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8099")), log_level="warning")


if __name__ == "__main__":
    main()
