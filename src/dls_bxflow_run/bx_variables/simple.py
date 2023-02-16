import logging

# Base class for a Thing which has a name and traits.
from dls_bxflow_api.thing import Thing

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.variables.simple"


class Simple(Thing):
    """
    Object representing a variable for a simple command.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)
