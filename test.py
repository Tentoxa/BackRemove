import os
import sys
import mimetypes
import requests

INPUT_IMAGE = r"C:\Users\Lukas\Downloads\ram.jpg"
OUTPUT_PATH = r"C:\Users\Lukas\Desktop\test_no_bg.png"
API_URL = "http://localhost:8080/remove-bg"
API_KEY = os.getenv("API_KEY")

if not os.path.exists(INPUT_IMAGE):
    print(f"Input image not found: {INPUT_IMAGE}")
    sys.exit(1)

headers = {}
if API_KEY:
    headers["X-API-Key"] = API_KEY

mime_type = mimetypes.guess_type(INPUT_IMAGE)[0] or "application/octet-stream"

print(f"Sending {INPUT_IMAGE} to {API_URL} ...")

with open(INPUT_IMAGE, "rb") as f:
    resp = requests.post(
        API_URL,
        files={"file": (os.path.basename(INPUT_IMAGE), f, mime_type)},
        headers=headers,
    )

if resp.status_code == 200:
    with open(OUTPUT_PATH, "wb") as out:
        out.write(resp.content)
    print(f"Saved to {OUTPUT_PATH}")
else:
    print(f"Error {resp.status_code}: {resp.text}")
    sys.exit(1)
