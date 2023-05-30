class ExecutionSummary:
    """
    Class for interacting with execution summary output.
    """

    def __init__(self):
        self.filename = "execution_summary"

    # -----------------------------------------------------------------
    def append_text(self, text: str) -> None:
        """
        Append text string to execution summary.

        Args:
            text (str): Text string to add to the file.
        """
        self.__append_raw(text)

    # -----------------------------------------------------------------
    def append_image(self, image: str) -> None:
        """
        Append image file base name to execution summary.

        Usually the bxflow task in a jupyter notebook or other class
        will call this when it knows there is an image to be displayed.

        Args:
            image (str): Name of image file.
                If no directory name, then it is assumed from the task directory.
        """
        if not image.startswith("/"):
            self.__append_raw("{bx_task_directory}/" + image)
        else:
            self.__append_raw(image)

    # -----------------------------------------------------------------
    def substitute_task_information(self, raw: str, bx_task) -> str:
        """
        Substitute task-specific information in the execution summary.

        Args:
            raw (str): Raw execution summary string.
                It may include subsitution symbols, such as {bx_task_directory}.
            bx_task (BxTask): Task object.
        """
        return raw.replace("{bx_task_directory}", bx_task.get_directory())

    # -----------------------------------------------------------------
    def compose_html(self, raw: str) -> str:
        """
        Compose an execution summary string into html.

        Makes divs out of lines in the summary raw text.

        Lines end with an image file type and are wrapped in an <img> tag.

        Args:
            raw (str): Raw execution summary string, possibly multiline separated by \n.
        """

        if raw is None:
            return ""

        lines = raw.split("\n")
        html = []
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            # Improve image mimetype recognition in ExecutionSummary.compose_as_html().
            if line.lower().endswith(".png") or line.lower().endswith(".jpg"):
                line = f"<img class='T_image' src='/filestore{line}' />"

            html.append(f"<div class='T_line'>{line}</div>")

        return "\n".join(html).strip()

    # -----------------------------------------------------------------
    def __append_raw(self, raw: str) -> None:
        """
        Append raw string to execution summary.

        Args:
            raw (str): Raw string to add to the file.
        """
        with open(self.filename, "at") as stream:
            stream.write(raw)
            stream.write("\n")
