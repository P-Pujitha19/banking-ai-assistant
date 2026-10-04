"""Exceptions raised by the mock banking layer."""
from __future__ import annotations


class BankError(Exception):
    """Base class for banking-domain errors; ``status_code`` mirrors HTTP semantics."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(BankError):
    """Requested customer / account / transaction / product does not exist."""

    status_code = 404


class InvalidRequestError(BankError):
    """The request parameters are invalid."""

    status_code = 422


class BankAPIError(Exception):
    """Raised by bank clients when the (mock) banking API returns an error."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message


class BankUnavailableError(BankAPIError):
    """The banking API could not be reached."""

    def __init__(self, message: str = "Banking service is unavailable") -> None:
        super().__init__(503, message)
