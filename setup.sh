#!/usr/bin/env bash
# One-time setup + launch for macOS/Linux:  bash setup.sh
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3.11+ first."; exit 1; }
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env - add your OPENAI_API_KEY to it, then re-run: bash setup.sh"
  exit 0
fi
streamlit run app.py
