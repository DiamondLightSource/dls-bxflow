import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
def data_label_2_filestore_directory(data_label):

    # Configurator interface.
    from dls_bxflow_lib.bx_configurators.bx_configurators import (
        bx_configurators_get_default,
    )

    beamline = bx_configurators_get_default().require("visit.beamline")
    visit_directory = bx_configurators_get_default().require("visit.directory")

    filestore_directory = f"{visit_directory}/processing/{beamline}-{data_label}"

    return filestore_directory
