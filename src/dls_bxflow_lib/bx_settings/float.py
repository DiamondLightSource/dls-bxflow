import logging

# Utilities.
from dls_utilpack.describe import describe

# Base class for this class.
from dls_bxflow_lib.bx_settings.base import Base

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_settings.float"


class Float(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.css_style = "T_float"

    # ----------------------------------------------------------------------------------------
    def set_value(self, value):
        """Convert provided value to proper type if possible"""

        if value is None:
            pass

        elif isinstance(value, str) and value.strip() == "":
            value = None

        else:
            try:
                value = float(value)
            except Exception:
                raise RuntimeError(f"value {value} cannot be interpreted as a float")

        logger.debug(describe(f"[SETTINGVAL] converted setting {self.uuid()}", value))

        Base.set_value(self, value)

    # ----------------------------------------------------------------------------------------
    def type_converter(string):
        """"""

        return float(string)

    # ----------------------------------------------------------------------------------------
    def get_type_name(self):
        """"""

        return "float"
