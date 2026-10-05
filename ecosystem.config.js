module.exports = {
  apps: [
    {
      name: "coilbrain",
      script: ".venv/Scripts/uvicorn.exe",
      args: "coilbrain.main:app --host 0.0.0.0 --port 8000",
      cwd: "./coilbrain",
      interpreter: "none",
      env: {
        "BYBIT_DEMO_API_KEY": "K5SmIqF12HmGzA3odL",
        "BYBIT_DEMO_API_SECRET": "RE65fU5z3hYmrEtEtxV6yPNza3hDYiGjOqZg",
        "PYTHONIOENCODING": "utf-8"
      }
    },
    {
      name: "coilspring_v2",
      script: "coiled_spring_scanner.py",
      cwd: "./",
      interpreter: "python",
      env: {
        "PYTHONIOENCODING": "utf-8",
        "COILBRAIN_URL": "http://localhost:8000/signals"
      }
    }
  ]
};
