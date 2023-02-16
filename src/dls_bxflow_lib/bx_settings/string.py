import logging

# Base class for this class.
from dls_bxflow_lib.bx_settings.base import Base

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_settings.string"


class String(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.css_style = "T_string"

    # ----------------------------------------------------------------------------------------
    def type_converter(string):
        """"""

        return string

    # ----------------------------------------------------------------------------------------
    def get_type_name(self):
        """"""

        return "string"
