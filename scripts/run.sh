#!/bin/bash
cd "$(dirname "$0")/.."
exec .venv/bin/python pepar/app.py
