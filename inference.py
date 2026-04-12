import os
import requests
import json

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

def query(payload):
    try:
        if not API_BASE_URL or not MODEL_NAME:
            return {"error": "Missing environment variables"}

        headers = {
            "Content-Type": "application/json"
        }

        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        response = requests.post(
            f"{API_BASE_URL}/{MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=30
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}
