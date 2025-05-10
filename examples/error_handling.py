"""Example demonstrating automatic error handling in autogenlib."""

from autogenlib import setup_exception_handler

# Initialize error handling
setup_exception_handler()

# Throw an error
1 / 0
