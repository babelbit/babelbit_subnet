import os

# Disable solo challenge in tests by default to avoid exercising unmocked solo paths.
os.environ.setdefault("BB_ENABLE_SOLO_CHALLENGE", "0")
