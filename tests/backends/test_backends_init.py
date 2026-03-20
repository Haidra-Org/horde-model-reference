"""Tests for the backends package __getattr__ lazy-import pattern."""

import pytest


def test_lazy_redis_import() -> None:
    """__getattr__('RedisBackend') should return the RedisBackend class."""
    import horde_model_reference.backends as backends_pkg

    cls = backends_pkg.__getattr__("RedisBackend")
    assert isinstance(cls, type)
    assert cls.__name__ == "RedisBackend"


def test_getattr_unknown_raises() -> None:
    """__getattr__ for an unknown name should raise AttributeError."""
    import horde_model_reference.backends as backends_pkg

    with pytest.raises(AttributeError, match="Bogus"):
        backends_pkg.__getattr__("Bogus")
