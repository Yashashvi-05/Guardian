"""CLI entry point for guardian-mcp proxy server."""
from __future__ import annotations

import argparse
import os
import sys

import yaml
import uvicorn

from guardian_mcp_proxy.proxy import create_app


def main():
    parser = argparse.ArgumentParser(
        description="GUARDIAN MCP Security Proxy",
        epilog="Example: guardian-mcp --config guardian_mcp.yaml --port 8000",
    )
    parser.add_argument("--config", default="guardian_mcp.yaml", help="Path to YAML config file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"[guardian-mcp] Config file {args.config!r} not found — using defaults")

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
