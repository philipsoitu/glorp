"""Single-worker local server entry point."""

import argparse

import uvicorn

from .app import create_app
from .config import ConfigError, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Glorp: local, transcript-recording OpenAI-compatible proxy")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration (default: config.yaml)")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    uvicorn.run(create_app(config), host=config.host, port=config.port, workers=1, access_log=False)
