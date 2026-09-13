"""Structured logging setup using loguru."""

import sys
from loguru import logger


def setup_logging(verbose: bool = False):
    """
    Configure loguru for the VisionEdit pipeline.

    Parameters
    ----------
    verbose : bool
        If True, sets log level to DEBUG. Otherwise INFO.

    Returns
    -------
    loguru.Logger
        The configured logger instance.
    """
    logger.remove()  # Remove default handler

    level = "DEBUG" if verbose else "INFO"

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    return logger
