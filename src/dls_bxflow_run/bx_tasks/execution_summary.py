class ExecutionSummary:
    """
    Static class for interacting with execution summary output.
    """

    filename = "execution_summary"

    def append_raw(raw: str) -> None:
        """
        Append raw string to execution summary.

        Args:
            raw (str): Raw string to add to the file.
        """
        with open(ExecutionSummary.filename, "at") as stream:
            stream.write(raw)
