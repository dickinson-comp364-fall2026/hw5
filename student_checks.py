"""Student-facing checks for incomplete assignment work."""

from __future__ import annotations

import inspect
from pathlib import Path


class IncompleteTodoError(RuntimeError):
    """Raised when a required assignment TODO has not been completed."""

    def __init__(self) -> None:
        caller = inspect.currentframe().f_back
        if caller is None:
            message = "A required TODO is not completed."
        else:
            file_name = Path(caller.f_code.co_filename).name
            line_number = caller.f_lineno
            scope = caller.f_code.co_qualname
            message = (
                f"The TODO near line {line_number} in {file_name} "
                f"({scope}) is not completed."
            )

        super().__init__(message)