import os

# src.config instantiates Settings at import time, which requires these keys.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("SPOONACULAR_API_KEY", "test-spoonacular-key")
