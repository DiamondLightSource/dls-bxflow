import logging

# Base class for generic things.
from dls_utilpack.thing import Thing

logger = logging.getLogger(__name__)


class Base(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None):
        Thing.__init__(self, thing_type, specification)
