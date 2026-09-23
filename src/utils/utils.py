import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str | Path,
    level: int = logging.INFO,
) -> logging.Logger:

    log_file = Path(log_file)

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = False

    # Prevent duplicate handlers.
    if logger.handlers:
        return logger

    handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s - "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger