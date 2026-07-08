"""
run_api.py

Entry point to start the FastAPI server.

Usage:
    python run_api.py

Or directly with uvicorn:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
