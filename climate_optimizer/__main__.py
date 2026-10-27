"""python -m climate_optimizer: run the service."""

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if os.environ.get("DEMO"):
        from .demo import demo_app
        app = demo_app()
    else:
        from .app import create_app
        app = create_app()
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8099")), log_level="warning")


if __name__ == "__main__":
    main()
