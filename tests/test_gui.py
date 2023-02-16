import logging

import yaml

# API constants.
from dls_servbase_api.constants import Keywords as ProtocoljKeywords

# Utilities.
from dls_utilpack.describe import describe

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing gui
from dls_bxflow_lib.bx_guis.bx_guis import bx_guis_get_default
from dls_bxflow_lib.bx_guis.constants import Commands, Cookies, Keywords

# Version.
from dls_bxflow_lib.version import version as dls_bxflow_lib_version

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestGuiLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        GuiTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class GuiTester(BaseContextTester):
    """
    Class to test the news.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        # Don't build the intermediate level servers.
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_catalog_specification")

        context_configuration = await bx_configurator.load()

        bx_gui_specification = context_configuration["bx_gui_specification"]
        type_specific_tbd = bx_gui_specification["type_specific_tbd"]
        aiohttp_specification = type_specific_tbd["aiohttp_specification"]
        aiohttp_specification["search_paths"] = [output_directory]

        bx_context = BxContexts().build_object(context_configuration)

        gaml_settings = {"current": {"section1": {"a": 1}}}
        gaml_filename = f"{output_directory}/gaml.yaml"
        with open(gaml_filename, "w") as stream:
            yaml.dump(gaml_settings, stream, default_flow_style=False, sort_keys=False)

        async with bx_context:

            # Use protocolj to fetch a request from the dataface.
            # Simulates what javascript would do by ajax.
            request = {
                Keywords.COMMAND: Commands.GET_RECENT_JOBS,
                ProtocoljKeywords.ENABLE_COOKIES: [Cookies.RECENT_JOBS_UX],
            }

            response = await bx_guis_get_default().client_protocolj(request)

            # The response is json, with "html" keyword containing the html string.
            assert "html" in response
            assert "<tbody>\n</tbody>" in response["html"]

            # --------------------------------------------------------------------
            # Now make a request which composes workflow settings.
            if True:
                request = {
                    Keywords.COMMAND: Commands.SHOW_WORKFLOW_SETTINGS,
                    ProtocoljKeywords.ENABLE_COOKIES: [Cookies.JOB_SUBMIT_UX],
                    "workflow_filename_classname": "tests/workflows/e/workflow.py::E",
                    "data_label": "my_data_label",
                    "job_label": "my_job_label",
                }

                response = await bx_guis_get_default().client_protocolj(
                    request, cookies={}
                )

                logger.debug(describe("[WORKSET] response", response))

                # The response is json, with "html" keyword containing the html string.
                assert "settings_html" in response

                # We should also have cookies back.
                assert "__cookies" in response
                cookies = response["__cookies"]
                assert Cookies.JOB_SUBMIT_UX in cookies

                # Use the cookie name in the next request.
                # cookie_uuid = cookies[Cookies.JOB_SUBMIT_UX]

            # --------------------------------------------------------------------
            # Write a text file and fetch it through the http server.
            filename = "test.html"
            contents = "some test html"
            with open(f"{output_directory}/{filename}", "wt") as file:
                file.write(contents)
            text = await bx_guis_get_default().client_get_file(filename)
            assert text == contents

            # Write a binary file and fetch it through the http server.
            filename = "test.exe"
            contents = "some test binary"
            with open(f"{output_directory}/{filename}", "wt") as file:
                file.write(contents)
            binary = await bx_guis_get_default().client_get_file(filename)
            # Binary comes back as bytes due to suffix of url filename.
            assert binary == contents.encode()

            # --------------------------------------------------------------------
            # Get an html file automatically configured in bx_guis/html.
            filename = "javascript/bxflow/version.js"

            contents = f' = "{dls_bxflow_lib_version()}"'
            # TODO: Figure out a way to put the package's version into the version.js file.
            contents = "TBD"
            text = await bx_guis_get_default().client_get_file(filename)
            logger.debug(f"javascript version is {text.strip()}")
            assert contents in text
