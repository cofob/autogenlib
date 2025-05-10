"""Automatic code generation library using OpenAI."""

import sys
from ._finder import AutoLibFinder


def init(desc):
    """Initialize autogenlib with a description of the functionality needed."""
    # Update the global description
    from . import _state

    _state.description = desc

    # Add our custom finder to sys.meta_path if it's not already there
    for finder in sys.meta_path:
        if isinstance(finder, AutoLibFinder):
            finder.description = desc  # Update the description in the existing finder
            return
    sys.meta_path.insert(0, AutoLibFinder(desc))
