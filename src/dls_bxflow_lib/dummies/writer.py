# Dummy class to write a file.
# Main program to cover it.

import argparse


class Writer:
    # -----------------------------------------------------------------------------
    def configure(self, configuration):
        self._configuration = configuration.copy()

    # -----------------------------------------------------------------------------
    async def run(self):
        filename = self._configuration.get("filename")

        if filename is None:
            raise RuntimeError("filename keyword is not provided in configuration")

        message = self._configuration.get("message")

        with open(filename, "w") as file:
            file.write(message)
            file.write("\n")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Make a parser.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filename",
        help="filename to write to",
        type=str,
        metavar="filename",
    )

    parser.add_argument(
        "message",
        help="message to write",
        type=str,
        metavar="string",
    )

    args = parser.parse_args()

    # Make an instance of the writer class.
    writer = Writer()

    # Configure the writer instance.
    writer.configure({"filename": args.filename, "message": args.message})

    # Let the writer do its bx_job.
    writer.run()
