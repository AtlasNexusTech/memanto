from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_moorcheh_for_tests():
    """Prevent tests from calling real Moorcheh APIs."""
    with (
        patch("memanto.app.clients.moorcheh.MoorchehClient") as mock_client_singleton,
    ):
        mock_instance = MagicMock()
        mock_instance.namespaces.create.return_value = {"status": "created"}
        mock_instance.namespaces.list.return_value = {"namespaces": []}
        mock_client_singleton.return_value = mock_instance
        yield mock_instance
