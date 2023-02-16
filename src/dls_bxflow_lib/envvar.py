import logging
import os

logger = logging.getLogger(__name__)


class Envvar:
    """Class which covers environment variables, with default values."""

    BXFLOW_CONFIGFILE = "BXFLOW_CONFIGFILE"
    BXFLOW_DATA = "BXFLOW_DATA"
    BXFLOW_DLS_ROOT = "BXFLOW_DLS_ROOT"
    BEAMLINE = "BEAMLINE"
    VISIT_YEAR = "VISIT_YEAR"
    VISIT = "VISIT"

    def __init__(self, name, **kwargs):

        environ = kwargs.get("environ")

        if environ is None:
            environ = os.environ

        self.name = name
        self.is_set = False
        self.value = None

        if name in environ:
            self.is_set = True
            self.value = environ[name]
        else:
            if "default" in kwargs:
                self.is_set = True
                self.value = kwargs["default"]
            else:
                self.is_set = False
                self.value = None
