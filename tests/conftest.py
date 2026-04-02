import os

# Disable solo challenge in tests by default to avoid exercising unmocked solo paths.
os.environ.setdefault("BB_ENABLE_SOLO_CHALLENGE", "0")
# Disable round2 challenge by default; tests enable it explicitly when needed.
os.environ.setdefault("BB_ENABLE_ARENA_CHALLENGE", "0")
