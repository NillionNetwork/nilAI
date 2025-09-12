"""Global pytest configuration."""

import asyncio
import warnings


def pytest_configure(config):
    """Configure pytest to suppress StreamWriter errors."""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Monkey patch StreamWriter.__del__ to suppress exceptions
    original_del = asyncio.StreamWriter.__del__

    def silent_del(self):
        try:
            original_del(self)
        except Exception:
            pass

    asyncio.StreamWriter.__del__ = silent_del
