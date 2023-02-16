import logging

import pytest

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Results settings.
from dls_bxflow_lib.bx_settings.bx_settings import BxSettings

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestSettings:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        SettingsTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class SettingsTester(BaseContextTester):
    """
    Class to test the settings.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:

            # -----------------------------------------------------------------
            # Add some settings.

            bx_settings1 = BxSettings("settings1")
            bx_settings2 = BxSettings("settings2")

            bx_settings1.add(
                bx_settings1.build_object(
                    {
                        "type": "dls_bxflow_lib.bx_settings.string",
                        "uuid": "setting1",
                        "value": "my \"good\" 'thing'",
                    }
                )
            )
            bx_settings1.add(
                bx_settings1.build_object(
                    {
                        "type": "dls_bxflow_lib.bx_settings.float",
                        "uuid": "setting2",
                        "value": 1.23,
                    }
                )
            )
            bx_settings1.add(
                bx_settings1.build_object(
                    {
                        "type": "dls_bxflow_lib.bx_settings.integer",
                        "uuid": "setting3",
                        "value": 123,
                    }
                )
            )
            bx_settings1.add(
                bx_settings1.build_object(
                    {
                        "type": "dls_bxflow_lib.bx_settings.boolean",
                        "uuid": "setting4",
                        "value": "1",
                    }
                )
            )
            bx_settings1.add(
                bx_settings1.build_object(
                    {
                        "type": "dls_bxflow_lib.bx_settings.string",
                        "uuid": "exclude_me",
                        "value": "the excluded thing",
                    }
                )
            )

            filename = f"{output_directory}/settings.yaml"

            bx_settings1.save(filename)

            bx_settings2.load(filename)

            html_string = bx_settings2.compose_html_inputs(exclusions=["exclude_me"])

            logger.debug(f"composed html inputs\n{html_string}")

            html_string = bx_settings2.compose_html_outputs(exclusions=["exclude_me"])

            logger.debug(f"composed html outputs\n{html_string}")

            python_string = bx_settings2.compose_python_dict_assignment("settings")

            logger.debug(f"composed python dict\n{python_string}")

            python_string = bx_settings2.compose_python_variable_assignment()
            python_string = "\n".join(python_string)

            logger.debug(f"composed python variables\n{python_string}")

            # ----------------------------------------------------------
            with pytest.raises(NotFound):
                bx_settings1.get_value("zzzz")

            with pytest.raises(NotFound):
                bx_settings1.set_value("zzzz", "1234")

            # ----------------------------------------------------------
            bx_settings3 = BxSettings("settings3")
            with pytest.raises(RuntimeError):
                bx_settings3.add(
                    bx_settings1.build_object(
                        {
                            "type": "dls_bxflow_lib.bx_settings.boolean",
                            "uuid": "setting1",
                            "value": "x",
                        }
                    )
                )

            # logger.debug("got expected error", exc_info=exception_info.value)

            # ----------------------------------------------------------
            with pytest.raises(RuntimeError):
                bx_settings3.add(
                    bx_settings1.build_object(
                        {
                            "type": "dls_bxflow_lib.bx_settings.integer",
                            "uuid": "setting2",
                            "value": "x",
                        }
                    )
                )

            # logger.debug("got expected error", exc_info=exception_info.value)

            # ----------------------------------------------------------
            with pytest.raises(RuntimeError):
                bx_settings3.add(
                    bx_settings1.build_object(
                        {
                            "type": "dls_bxflow_lib.bx_settings.float",
                            "uuid": "setting3",
                            "value": "x",
                        }
                    )
                )

            # logger.debug("got expected error", exc_info=exception_info.value)
