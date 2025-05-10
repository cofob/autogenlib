"""Automatic code generation library using OpenAI."""

import sys
from ._finder import AutoLibFinder
from ._exception_handler import setup_exception_handler


def init(desc, enable_exception_handler=True):
    """Initialize autogenlib with a description of the functionality needed.

    Args:
        desc (str): A description of the library you want to generate.
        enable_exception_handler (bool): Whether to enable the global exception handler
            that sends exceptions to LLM for fix suggestions. Default is True.
    """
    # Update the global description
    from . import _state

    _state.description = desc
    _state.exception_handler_enabled = enable_exception_handler

    # Set up exception handler if enabled
    if enable_exception_handler:
        from ._exception_handler import setup_exception_handler

        setup_exception_handler()

    # Add our custom finder to sys.meta_path if it's not already there
    for finder in sys.meta_path:
        if isinstance(finder, AutoLibFinder):
            finder.description = desc  # Update the description in the existing finder
            return
    sys.meta_path.insert(0, AutoLibFinder(desc))


def set_exception_handler(enabled=True):
    """Enable or disable the exception handler.

    Args:
        enabled (bool): Whether to enable the exception handler. Default is True.
    """
    from . import _state

    _state.exception_handler_enabled = enabled


__all__ = ["init", "set_exception_handler", "setup_exception_handler"]
