# CoilBrain

> Trading signal analysis service for the CoilSpring ecosystem.

CoilBrain receives signals from the CoilSpring bot, applies risk controls, and
will eventually execute trades through the Bybit Demo API.

## Project layout

```
coilbrain/
├── src/
│   └── coilbrain/
│       ├── __init__.py   # package version
│       ├── main.py       # FastAPI app + routes
│       ├── config.py     # pydantic-settings configuration
│       └── models.py     # shared Pydantic response models
├── tests/
│   └── test_health.py
├── .env.example          # template — copy to .env
├── .gitignore
├── pyproject.toml        # build + pytest config (src layout)
├── requirements.txt
└── README.md
```

## Quick start

### 1 — Create & activate a virtual environment

```powershell
# From the coilbrain/ directory
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .          # makes the src/ layout importable
```

### 3 — Configure environment (optional)

```powershell
Copy-Item .env.example .env
# Edit .env with your preferred values
```

### 4 — Run the development server

```powershell
uvicorn coilbrain.main:app --reload
```

Server starts at <http://localhost:8000>.  
Interactive API docs: <http://localhost:8000/docs>

### 5 — Run the tests

```powershell
pytest
```

## Endpoints

| Method | Path      | Description      |
|--------|-----------|------------------|
| GET    | `/health` | Liveness check   |
| GET    | `/docs`   | Swagger UI       |
| GET    | `/redoc`  | ReDoc UI         |

## Environment variables

| Variable       | Default      | Description                     |
|----------------|--------------|---------------------------------|
| `APP_HOST`     | `0.0.0.0`    | Bind address                    |
| `APP_PORT`     | `8000`       | Bind port                       |
| `DEBUG`        | `false`      | Enable uvicorn auto-reload      |
| `LOG_LEVEL`    | `INFO`       | Python logging level            |
| `SERVICE_NAME` | `coilbrain`  | Reported in /health response    |
