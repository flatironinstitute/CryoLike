import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-debug-tests",
        action="store_true",
        default=False,
        help="Run debug tests (e.g. verbose output, printing plots)"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "debug_test: mark test as a verbose-output debug test"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-debug-tests"):
        return
    skip_debug = pytest.mark.skip(
        reason="requires --run-debug-tests option to run"
    )
    for item in items:
        if "debug_test" in item.keywords:
            item.add_marker(skip_debug)
