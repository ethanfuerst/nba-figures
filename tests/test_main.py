import runpy
from unittest.mock import patch

import pytest


def test_main_module():
    with patch('nbafigs.cli.main') as mock_main:
        mock_main.side_effect = SystemExit(0)
        with pytest.raises(SystemExit):
            runpy.run_module('nbafigs', run_name='__main__')

        mock_main.assert_called_once()
