import numpy as np
import pytest

from mojo_statsmodels_tsa._lib import addr


def test_addr_rejects_buffers_outside_the_ffi_contract():
    with pytest.raises(TypeError, match="float64"):
        addr(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((3, 3), dtype=np.float64)[:, ::2])
    with pytest.raises(ValueError, match="empty"):
        addr(np.empty(0, dtype=np.float64))


def test_addr_requires_writable_output():
    values = np.ones(3, dtype=np.float64)
    values.flags.writeable = False
    assert addr(values) != 0
    with pytest.raises(ValueError, match="writable"):
        addr(values, writable=True)
