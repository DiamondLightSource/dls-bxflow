# Use standard logging in this module.
import logging
import os
from typing import Optional

# Use dls_servbase.
from dls_servbase_lib.configurators.configurators import (
    dls_servbase_configurators_set_default,
)

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

# Environment variables with some extra functionality.
from dls_bxflow_lib.envvar import Envvar

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_configurator = None


def bx_configurators_set_default(bx_configurator):
    global __default_bx_configurator
    __default_bx_configurator = bx_configurator

    # Since bx uses dls_servbase to launch processes.
    dls_servbase_configurators_set_default(bx_configurator)


def bx_configurators_get_default():
    global __default_bx_configurator
    if __default_bx_configurator is None:
        raise RuntimeError("bx_configurators_get_default instance is None")
    return __default_bx_configurator


def bx_configurators_has_default():
    global __default_bx_configurator
    return __default_bx_configurator is not None


# -----------------------------------------------------------------------------------------
class BxConfigurators(Things):
    """
    Configuration loader.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        bx_configurator_class = self.lookup_class(
            require(f"{callsign(self)} specification", specification, "type")
        )

        try:
            bx_configurator_object = bx_configurator_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to instantiate bx_configurator object from module %s"
                % (bx_configurator_class.__module__)
            ) from exception

        return bx_configurator_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_configurators.yaml":
            from dls_bxflow_lib.bx_configurators.yaml import Yaml

            return Yaml

        raise NotFound("unable to get bx_configurator class for type %s" % (class_type))

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
                "type": "dls_bxflow_lib.bx_configurators.yaml",
                "type_specific_tbd": {"filename": configurator_filename},
            }
        )

        bx_configurator.substitute(
            {"configurator_directory": os.path.dirname(configurator_filename)}
        )

        return bx_configurator
