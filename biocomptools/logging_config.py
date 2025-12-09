import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Optional, Dict, Union

from rich.logging import RichHandler
from biocomptools.toollib.config import config

_logging_setup_done = False


class SafeRichHandler(RichHandler):
    """RichHandler that falls back to basic stderr output on style corruption errors.

    Works around a rare issue where rich's Style._color can become corrupted
    (e.g., to a tuple instead of Color object) after multiprocessing, causing
    AttributeError: 'tuple' object has no attribute 'downgrade'
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except AttributeError as e:
            if "downgrade" in str(e) or "_color" in str(e):
                # Fall back to basic output on style corruption
                try:
                    msg = self.format(record)
                    sys.stderr.write(f"{msg}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            else:
                raise


class LevelFilter(logging.Filter):
    """
    A filter that only allows records whose level is at least the effective level
    of the logger that issued them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        effective_level = logging.getLogger(record.name).getEffectiveLevel()
        return record.levelno >= effective_level


def setup_logging(
    log_file: Optional[Path] = None,
    logger_levels: Optional[Dict[str, Union[str, int]]] = None,
    force: bool = False,
) -> None:
    """Configure logging using our config settings, with added prints and applying levels to children."""
    global _logging_setup_done

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    log_config = config.logging

    # Use the default_level from config for the root logger.
    if isinstance(log_config.default_level, str):
        default_level = getattr(logging, log_config.default_level.upper(), logging.INFO)
    else:
        default_level = log_config.default_level

    root_logger.setLevel(default_level)

    # Determine if we're in a worker process.
    is_worker = multiprocessing.current_process().name != "MainProcess"

    # Create the appropriate handler.
    if log_config.use_rich_handler and not is_worker:
        handler = SafeRichHandler(
            show_time=log_config.show_time,
            show_path=log_config.show_path,
            show_level=log_config.show_level,
            enable_link_path=log_config.enable_link_path,
            rich_tracebacks=log_config.rich_tracebacks,
            omit_repeated_times=log_config.omit_repeated_times,
            markup=log_config.markup,
            log_time_format=log_config.date_format,
        )
        formatter = logging.Formatter(log_config.rich_format, datefmt=log_config.date_format)
    else:
        handler = logging.StreamHandler()
        fmt = log_config.worker_format if is_worker else log_config.file_format
        formatter = logging.Formatter(fmt, datefmt=log_config.date_format)

    handler.setFormatter(formatter)
    # Set handler level to NOTSET so that filtering is done at the logger level.
    handler.setLevel(logging.NOTSET)
    # Add our custom filter.
    handler.addFilter(LevelFilter())
    root_logger.addHandler(handler)

    # Optionally add a file handler if a log_file is provided and not a worker process.
    if log_file and not is_worker:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(log_config.file_format, datefmt=log_config.date_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.NOTSET)
        file_handler.addFilter(LevelFilter())
        root_logger.addHandler(file_handler)

    # Prepare logger levels from config.
    levels_to_apply = {
        name: getattr(logging, level.upper()) if isinstance(level, str) else level
        for name, level in log_config.levels.items()
    }
    if logger_levels:
        # Override or add any logger levels provided as argument.
        levels_to_apply.update(
            {
                name: getattr(logging, level.upper()) if isinstance(level, str) else level
                for name, level in logger_levels.items()
            }
        )

    # Apply levels to any logger whose name matches or starts with the given key.
    for logger_key, level in levels_to_apply.items():
        # Iterate over all known loggers.
        for existing_logger in list(logging.root.manager.loggerDict.keys()):
            if existing_logger == logger_key or existing_logger.startswith(logger_key + "."):
                logging.getLogger(existing_logger).setLevel(level)
        # Also set the level on the base logger if it hasn't been created yet.
        logging.getLogger(logger_key).setLevel(level)

    _logging_setup_done = True


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Get a logger with the specified name and optional level."""
    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level_val = getattr(logging, level.upper(), logging.INFO)
        else:
            level_val = level
        logger.setLevel(level_val)

    return logger


def print_logger_hierarchy(logger_name: str):
    """Utility function to print the full hierarchy of a logger with debug prints."""
    print(f"\n=== Logger Hierarchy for {logger_name} ===")
    logger = logging.getLogger(logger_name)
    current = logger
    while current:
        print(f"Logger: {current.name}")
        print(f"Level: {logging.getLevelName(current.level)}")
        print(f"Effective Level: {logging.getLevelName(current.getEffectiveLevel())}")
        print(f"Propagate: {current.propagate}")
        print(f"Handlers: {len(current.handlers)}")
        current = current.parent
        if current:
            print("\nParent →")
