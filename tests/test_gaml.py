import logging

# Gaml manager.
from dls_bxflow_lib.bx_gamls.bx_gamls import BxGamls

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)

bx_job_uuid = "bx_job_uuid-0001"


# ----------------------------------------------------------------------------------------
class TestGamlBare:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = None
        GamlTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class GamlTester(BaseContextTester):
    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_gaml_specification = {
            "type": "dls_bxflow_lib.bx_gamls.html",
        }

        bx_gaml = BxGamls().build_object(bx_gaml_specification)

        contents = {
            "top_thing": "z",
            "section1": {"input1": "x", "section2": {"input2": "y"}},
        }

        # Compose html formw with input fields from the contents.
        html = bx_gaml.compose(contents)

        assert "name='top_thing'" in html
        assert "name='section1.input1'" in html
        assert "name='section1.section2.input2'" in html
        logger.debug("html\n%s" % (html))

        # Make a payload like what would be submitted from the html.
        payload = {"section1.input1": "x", "section1.section2.input2": "y"}

        # Parse the dotted-name syntax into a normal dict.
        contents = bx_gaml.parse(payload)
        # logger.debug("parsed contents\n%s" % (json.dumps(contents, indent=2)))

        assert contents["section1"]["input1"] == "x"
        assert contents["section1"]["section2"]["input2"] == "y"
