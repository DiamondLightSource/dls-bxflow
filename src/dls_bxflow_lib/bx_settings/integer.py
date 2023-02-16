import logging

# Utilities.
from dls_utilpack.describe import describe

# Base class for this class.
from dls_bxflow_lib.bx_settings.base import Base
from dls_bxflow_lib.bx_settings.constants import Types as BxSettingTypes

logger = logging.getLogger(__name__)

thing_type = BxSettingTypes.INTEGER


class Integer(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.css_style = "T_integer"

    # ----------------------------------------------------------------------------------------
    def set_value(self, value):
        """Convert provided value to proper type if possible."""

        if value is None:
            pass

        elif isinstance(value, str) and value.strip() == "":
            value = None

        else:
            try:
                value = int(value)
            except Exception:
                raise RuntimeError(f"value {value} cannot be interpreted as an integer")

        logger.debug(describe(f"[SETTINGVAL] converted setting {self.uuid()}", value))

        Base.set_value(self, value)

    # ----------------------------------------------------------------------------------------
    def type_converter(string):
        """"""

        return int(string)

    # ----------------------------------------------------------------------------------------
    def get_type_name(self):
        """"""

        return "integer"
