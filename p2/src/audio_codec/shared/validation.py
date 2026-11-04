#Contains the shared validation issue type used to collect all errors in one pass.
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationIssue:
    '''
    Holds information about a single validation failure.

    Attributes:
        - field (str): Name of the offending field.
        - message (str): Description of the validation failure.
    '''

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"
