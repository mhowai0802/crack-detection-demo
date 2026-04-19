"""``python -m backend`` → run uvicorn with the configured host/port."""

from __future__ import annotations

import uvicorn

from backend import config


def main() -> None:
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
