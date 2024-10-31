import logging
from rich.logging import RichHandler
from typing import Optional, Dict
from pathlib import Path
import os

DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure default levels for various loggers
DEFAULT_LOGGER_LEVELS: Dict[str, int] = {
    # External libraries
    'matplotlib': logging.WARNING,
    'matplotlib.font_manager': logging.ERROR,  # Suppress font debug messages
    'PIL': logging.WARNING,
    'jax': logging.WARNING,
    'ray': logging.WARNING,
    'fontTools': logging.WARNING,
    'h5py': logging.WARNING,
    'numba': logging.WARNING,
    'parso': logging.WARNING,
    # Project-specific default levels
    'biocomp': logging.ERROR,
    'biocomptools': logging.INFO,
    'biocomptools.plot': logging.INFO,
    'dracon': logging.INFO,
}


def setup_logging(
    default_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    logger_levels: Optional[Dict[str, int]] = None,
) -> None:
    """Configure logging for the biocomptools project.

    Args:
        default_level: Default logging level for all loggers
        log_file: Optional file path to write logs to
        logger_levels: Optional dict to override default logger levels

    logger_levels can also be set through the environment variable `BIOCOMP_LOGLEVEL_pkg_name=lvl`
    """

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup handlers
    handlers = []
    console_handler = RichHandler(
        show_path=True, omit_repeated_times=False, log_time_format=DEFAULT_DATE_FORMAT
    )
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
        handlers.append(file_handler)

    # Configure root logger
    root_logger.setLevel(default_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # Apply logger-specific levels
    levels_to_apply = DEFAULT_LOGGER_LEVELS.copy()
    if logger_levels:
        levels_to_apply.update(logger_levels)

    # Override levels from environment variables
    for env_var, level in os.environ.items():
        if env_var.startswith('BIOCOMP_LOGLEVEL_'):
            logger_name = env_var.split('BIOCOMP_LOGLEVEL_')[1].replace('_', '.')
            # update all loggers that start with the specified name
            for logger in logging.Logger.manager.loggerDict:
                if logger.startswith(logger_name):
                    levels_to_apply[logger] = getattr(logging, level.upper())
            # levels_to_apply[logger_name] = getattr(logging, level.upper())

    for logger_name, level in levels_to_apply.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the specified name and optional level."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
