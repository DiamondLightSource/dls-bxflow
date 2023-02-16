import logging

# Base class for a BxGate.
from dls_bxflow_run.bx_gates.base import Base

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_gates.standard"


class Standard(Base):
    """
    Object representing a standard bx_gate.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)
