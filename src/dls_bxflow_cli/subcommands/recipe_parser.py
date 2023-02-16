import glob

# Use standard logging in this module.
import logging

# Base class for femtocheck subcommands.
from dls_bxflow_cli.subcommands.base import Base

# Recipe parser.
from dls_bxflow_lib.recipe_parser.overall import Overall
from dls_bxflow_lib.recipe_parser.parser1 import Parser1

logger = logging.getLogger()


# --------------------------------------------------------------
class RecipeParser(Base):
    """
    Check a previously recorded file.
    """

    def __init__(self, args):
        super().__init__(args)

    # ----------------------------------------------------------
    def run(self):
        """"""

        filenames = glob.glob(self._args.recipe_filename, recursive=True)

        overall = Overall()

        for filename in filenames:
            logger.info(" ")

            # Make a reader who can grok this scan type.
            recipe_parser = Parser1(overall, filename)

            recipe_parser.parse()

            logger.info(recipe_parser.compose_as_prettytable())

        logger.info(overall.compose_as_prettytable())

    # ----------------------------------------------------------
    def add_arguments(parser):

        # Add positional arguments.
        parser.add_argument(
            help="recipe_filename",
            type=str,
            metavar="filename",
            dest="recipe_filename",
        )

        return parser
