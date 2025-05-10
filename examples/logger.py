from logging import DEBUG, basicConfig

basicConfig(level=DEBUG)

from autogenlib import init

init("Library for easy and beatiful logging in CLI")

from autogenlib.easylog import warn, info

warn("Ahtung!")
info("Some useful (no) message")
