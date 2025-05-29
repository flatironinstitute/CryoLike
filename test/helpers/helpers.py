from unittest.mock import Mock

def mock_ctxt(ret):
    mock_ctxt = Mock()
    mock_ctxt.__enter__ = Mock(return_value=ret)
    mock_ctxt.__exit__ = Mock(return_value=False)
    return mock_ctxt