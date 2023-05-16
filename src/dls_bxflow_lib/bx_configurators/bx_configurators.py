# Use standard logging in this module.
import logging
import os
from typing import Optional

# Configurator.
from dls_multiconf_lib.constants import ThingTypes as MulticonfThingTypes
from dls_multiconf_lib.multiconfs import (
    Multiconfs,
    multiconfs_get_default,
    multiconfs_has_default,
    multiconfs_set_default,
)

# Environment variables with some extra functionality.
from dls_bxflow_lib.envvar import Envvar

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_configurator = None


def bx_configurators_set_default(bx_configurator):
    multiconfs_set_default(bx_configurator)


def bx_configurators_get_default():
    return multiconfs_get_default()


def bx_configurators_has_default():
    return multiconfs_has_default()


# -----------------------------------------------------------------------------------------
class BxConfigurators(Multiconfs):
    """
    Configuration loader.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Multiconfs.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object_from_environment(
        self, environ: Optional[dict] = None, args_dict: Optional[dict] = None
    ):

        configuration_keyword = "configuration"

        configurator_filename = None

        if args_dict is not None:
            configurator_filename = args_dict.get(configuration_keyword)

        if configurator_filename is not None:
            # Make sure the path exists.
            if not os.path.exists(configurator_filename):
                raise RuntimeError(
                    f"unable to find --{configuration_keyword} file {configurator_filename}"
                )
        else:
            # Get the explicit name of the config file.
            bxflow_configfile = Envvar(
                Envvar.BXFLOW_CONFIGFILE,
                environ=environ,
            )

            # Config file is explicitly named?
            if bxflow_configfile.is_set:
                # Make sure the path exists.
                configurator_filename = bxflow_configfile.value
                if not os.path.exists(configurator_filename):
                    raise RuntimeError(
                        f"unable to find {Envvar.BXFLOW_CONFIGFILE} {configurator_filename}"
                    )
            # Config file is not explicitly named?
            else:
                raise RuntimeError(
                    f"command line --{configuration_keyword} not given and environment variable {Envvar.BXFLOW_CONFIGFILE} is not set"
                )

        bx_configurator = self.build_object(
            {
                "type": MulticonfThingTypes.YAML,
                "type_specific_tbd": {"filename": configurator_filename},
            }
        )

        bx_configurator.substitute(
            {"configurator_directory": os.path.dirname(configurator_filename)}
        )

        return bx_configurator
