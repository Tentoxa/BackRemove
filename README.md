# BackRemove

Background removal API using [WithoutBG](https://github.com/withoutbg/withoutbg) Focus.

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Or with Docker:

```bash
docker compose up --build
```

Model weights (~320 MB) are downloaded automatically on first start.

## Usage

```bash
curl -X POST http://localhost:8080/remove-bg \
  -F "file=@photo.jpg" \
  --output no-bg.png
```

Supported formats: JPEG, PNG, WebP, GIF, AVIF, SVG.

## Auth

Optional. Set the `API_KEY` environment variable to require an `X-API-Key` header on requests. Unset = open access.

```bash
API_KEY=my-secret uvicorn app.main:app --host 0.0.0.0 --port 8080
curl -H "X-API-Key: my-secret" -F "file=@photo.jpg" http://localhost:8080/remove-bg --output no-bg.png
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/remove-bg` | Optional | Remove background, returns PNG |
