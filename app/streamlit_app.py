"""Streamlit entry point; adjacent requirements keep deployment lightweight."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app import main

main()
