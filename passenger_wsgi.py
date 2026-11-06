"""Passenger entry point; keep deployment secrets in the server-local .env."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / ".vendor"))
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=False)

from application import application
