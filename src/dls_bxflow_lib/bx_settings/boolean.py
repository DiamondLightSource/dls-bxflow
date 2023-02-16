import logging

# Utilities.
from dls_utilpack.describe import describe

# Base class for this class.
from dls_bxflow_lib.bx_settings.base import Base

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_settings.boolean"


class Boolean(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        self.css_style = "T_boolean"

        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

    # ----------------------------------------------------------------------------------------
    def set_value(self, value):
        """Convert provided value to proper type if possible"""

        if isinstance(value, bool):
            pass

        elif isinstance(value, str):
            trues = ["1", "y", "yes", "t", "true"]
            falses = ["", "0", "n", "no", "f", "false"]
            v = value.lower().strip()
            if v in trues:
                value = True
            elif v in falses:
                value = False
            else:
                raise RuntimeError(f"value {value} is not one of {trues} or {falses}")

        elif isinstance(value, int):
            value = False if value == 0 else True

        else:
            raise RuntimeError(
                f"value {value} type {type(value).__name__} cannot be converted to boolean"
            )

        logger.debug(describe(f"[SETTINGVAL] converted setting {self.uuid()}", value))

        Base.set_value(self, value)

    # ----------------------------------------------------------------------------------------
    def type_converter(string):
        """"""

        return bool(string)

    # ----------------------------------------------------------------------------------------
    def get_type_name(self):
        """"""

        return "boolean"
