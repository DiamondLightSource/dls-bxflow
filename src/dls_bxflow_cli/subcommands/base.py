import logging
import os
import sys
import tempfile
from typing import Optional

# Utilities.
from dls_utilpack.visit import get_visit_year

# Configurator.
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    BxConfigurators,
    bx_configurators_set_default,
)

logger = logging.getLogger(__name__)


class Base:
    """
    Base class for femtocheck subcommands.  Handles details like configuration.
    """

    def __init__(self, args):
        self._args = args

        self.__temporary_directory = None

    # ----------------------------------------------------------------------------------------
    def get_bx_configurator(self, args_dict: Optional[dict] = None):

        bx_configurator = BxConfigurators().build_object_from_environment(
            args_dict=args_dict
        )

        # For convenience, make a temporary directory for this test.
        self.__temporary_directory = tempfile.TemporaryDirectory()

        # Make the temporary directory available to the configurator.
        bx_configurator.substitute(
            {"temporary_directory": self.__temporary_directory.name}
        )

        substitutions = {
            # APPS, aka softwaredir, is set at Diamond by module load global/directories.
            "APPS": os.environ.get("softwaredir", "softwaredir"),
            "sys.prefix": sys.prefix,
            "CWD": os.getcwd(),
            "HOME": os.environ.get("HOME", "HOME"),
            "USER": os.environ.get("USER", "USER"),
            "PATH": os.environ.get("PATH", "PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "PYTHONPATH"),
        }

        if hasattr(self._args, "visit") and self._args.visit != "VISIT":
            BEAMLINE = os.environ.get("BEAMLINE")
            if BEAMLINE is None:
                raise RuntimeError("BEAMLINE environment variable is not defined")
            year = get_visit_year(BEAMLINE, self._args.visit)
            substitutions["BEAMLINE"] = BEAMLINE
            substitutions["VISIT"] = self._args.visit
            substitutions["YEAR"] = year

        bx_configurator.substitute(substitutions)

        # Set this as the default configurator so it is available everywhere.
        bx_configurators_set_default(bx_configurator)

        return bx_configurator
