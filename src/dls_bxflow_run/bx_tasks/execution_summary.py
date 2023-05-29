class ExecutionSummary:
    """
    Class for interacting with execution summary output.
    """

    def __init__(self):
        self.filename = "execution_summary"

    def append_raw(self, raw: str) -> None:
        """
        Append raw string to execution summary.

        Args:
            raw (str): Raw string to add to the file.
        """
        with open(self.filename, "at") as stream:
            stream.write(raw)
