# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
OpenTSLM Global Logger

This module provides a centralized logging system for the OpenTSLM project.
The logger can be used across all modules to provide consistent logging with
verbose mode control.
"""

import logging
from typing import Optional


class OpenTSLMLogger:
    """Global logger class for OpenTSLM project with verbose mode control."""

    _instance = None
    _initialized = False

    def __new__(cls, verbose: bool = False):
        if cls._instance is None:
            cls._instance = super(OpenTSLMLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, verbose: bool = False):
        if not self._initialized:
            self.verbose = verbose
            self.logger = logging.getLogger("OpenTSLM")

            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

            if verbose:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.WARNING)
            self._initialized = True
        else:
            # Update verbose setting if already initialized
            self.verbose = verbose
            if verbose:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.WARNING)

    def info(self, message: str):
        """Log info message if verbose mode is enabled."""
        self.logger.info(f"{message}")

    def warning(self, message: str):
        """Log warning message (always shown)."""
        self.logger.warning(f"⚠️  {message}")

    def error(self, message: str):
        """Log error message (always shown)."""
        self.logger.error(f"❌ {message}")

    def debug(self, message: str):
        """Log debug message if verbose mode is enabled."""
        self.logger.debug(f"🔍 {message}")

    def success(self, message: str):
        """Log success message with checkmark icon."""
        self.logger.info(f"✅ {message}")

    def loading(self, message: str):
        """Log loading message with spinner icon."""
        self.logger.info(f"🔄 {message}")

    def rocket(self, message: str):
        """Log rocket message for exciting progress."""
        self.logger.info(f"🚀 {message}")

    def data(self, message: str):
        """Log data message for dataset related info."""
        self.logger.info(f"📈 {message}")

    def set_verbose(self, verbose: bool):
        """Update verbose setting."""
        self.verbose = verbose
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)


# Global logger instance
_global_logger: Optional[OpenTSLMLogger] = None
_global_verbose_setting: bool = False


def get_logger(verbose: bool = None) -> OpenTSLMLogger:
    """
    Get the global OpenTSLM logger instance.

    Args:
        verbose: Whether to enable verbose logging. If None, uses global setting.

    Returns:
        OpenTSLMLogger instance
    """
    global _global_logger, _global_verbose_setting
    if _global_logger is None:
        # If no global logger exists, create one with the provided verbose setting
        # or use global setting if no verbose setting provided
        if verbose is None:
            verbose = _global_verbose_setting
        _global_logger = OpenTSLMLogger(verbose=verbose)
    else:
        # If global logger exists, update its verbose setting
        if verbose is not None:
            _global_logger.set_verbose(verbose)
        else:
            # If no verbose parameter provided, use global setting
            _global_logger.set_verbose(_global_verbose_setting)
    return _global_logger


def set_global_verbose(verbose: bool):
    """
    Set the global verbose setting for all logger instances.

    Args:
        verbose: Whether to enable verbose logging
    """
    global _global_logger, _global_verbose_setting
    _global_verbose_setting = verbose
    if _global_logger is None:
        _global_logger = OpenTSLMLogger(verbose=verbose)
    else:
        _global_logger.set_verbose(verbose)
