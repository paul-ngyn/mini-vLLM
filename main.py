"""Entry point: `python main.py` starts the server."""

import uvicorn

import config

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
