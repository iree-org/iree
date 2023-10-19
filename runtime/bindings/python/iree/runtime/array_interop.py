# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""BufferView and Python Array Protocol interop."""

from typing import Optional, Tuple
import logging
import numpy as np
import numpy.lib.mixins

from ._binding import (
    BufferUsage,
    HalBufferView,
    HalDevice,
    HalElementType,
    MappedMemory,
    MemoryType,
    HalFence,
)

__all__ = [
    "asdevicearray",
    "DeviceArray",
]

_DEVICE_HANDLED_FUNCTIONS = {}


def _device_implements(np_function):
    """Decorator that registers a base class implementation."""

    def decorator(func):
        _DEVICE_HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class DeviceArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    """An IREE device array.

    Device arrays can be in one of two states:
      1. Host accessible: The array will be backed by host accessible memory
         and can have the usual things done with it that one expects to be
         able to do with an ndarray.
      2. Device resident: The array is just a handle to a device resident
         Buffer (and BufferView wrapper). Metadata about the array are accessible
         (shape and dtype) but anything that touches the data cannot be accessed
         in this state.

    How a device array comes into existence controls how it can transition
    between these states:
      * A user can create a DeviceArray explicitly with a device allocator.
        Such an array will not be implicitly convertible to host accessible,
        although accessors exist to do so.
      * When created by the platform with a synchronization policy, then
        implicit transfer back to the host will trigger appropriate waits and
        be performed automatically (this is the common case for function return
        values if not otherwise configured, as an example).
    """

    def __init__(
        self,
        device: HalDevice,
        buffer_view: HalBufferView,
        implicit_host_transfer: bool = False,
        override_dtype=None,
    ):
        self._device = device
        self._buffer_view = buffer_view
        self._implicit_host_transfer = implicit_host_transfer
        self._override_dtype = override_dtype

        # If the array is host accessible, these will be non-None.
        self._mapped_memory: Optional[MappedMemory] = None
        self._host_array: Optional[np.ndarray] = None

    def __array__(self, dtype=None):
        self._transfer_to_host(True)
        if dtype is None:
            return self._host_array
        else:
            return self._host_array.__array__(dtype)  # pytype: disable=attribute-error

    def __array_function__(self, func, types, args, kwargs):
        if func in _DEVICE_HANDLED_FUNCTIONS:
            return _DEVICE_HANDLED_FUNCTIONS[func](*args, **kwargs)

        # Anything else forces a transfer to host and then delegates to the
        # host array.
        host_array = self.to_host()
        return host_array.__array_function__(
            func, types, args, kwargs
        )  # pytype: disable=attribute-error

    def __repr__(self):
        return f"<IREE DeviceArray: shape={np.shape(self)}, dtype={self.dtype}>"

    @property
    def is_host_accessible(self):
        """Whether this array is currently host accessible."""
        return self._host_array is not None

    def to_host(self) -> np.ndarray:
        self._transfer_to_host(False)
        return self._host_array

    def _is_mappable(self) -> bool:
        buffer = self._buffer_view.get_buffer()
        if (
            buffer.memory_type() & int(MemoryType.HOST_VISIBLE)
            != MemoryType.HOST_VISIBLE
        ):
            return False
        if (
            buffer.allowed_usage() & int(BufferUsage.MAPPING_SCOPED)
            != BufferUsage.MAPPING_SCOPED
        ):
            return False
        return True

    def _transfer_to_host(self, implicit):
        if self._host_array is not None:
            return
        if implicit and not self._implicit_host_transfer:
            raise ValueError(
                "DeviceArray cannot be implicitly transferred to the host: "
                "if necessary, do an explicit transfer via .to_host()"
            )
        if self._is_mappable():
            self._mapped_memory, self._host_array = self._map_to_host()
        else:
            self._host_array = self._copy_to_host()

    def _map_to_host(self) -> Tuple[MappedMemory, np.ndarray]:
        # TODO: When synchronization is enabled, need to block here.
        raw_dtype = self._get_raw_dtype()
        mapped_memory = self._buffer_view.map()
        host_array = mapped_memory.asarray(self._buffer_view.shape, raw_dtype)
        # Detect if we need to force an explicit conversion. This happens when
        # we were requested to pretend that the array is in a specific dtype,
        # even if that is not representable on the device. You guessed it:
        # this is to support bools.
        if self._override_dtype is not None and self._override_dtype != raw_dtype:
            host_array = host_array.astype(self._override_dtype)
        return mapped_memory, host_array

    def _copy_to_host(self) -> np.ndarray:
        # TODO: When synchronization is enabled, need to block here.
        source_buffer = self._buffer_view.get_buffer()
        host_buffer = self._device.allocator.allocate_buffer(
            memory_type=(MemoryType.HOST_LOCAL | MemoryType.DEVICE_VISIBLE),
            allowed_usage=(BufferUsage.TRANSFER_TARGET | BufferUsage.MAPPING_SCOPED),
            allocation_size=source_buffer.byte_length(),
        )
        # Copy and wait for buffer to be copied from source buffer.
        sem = self._device.create_semaphore(0)
        self._device.queue_copy(
            source_buffer,
            host_buffer,
            wait_semaphores=HalFence.create_at(sem, 0),
            signal_semaphores=HalFence.create_at(sem, 1),
        )
        HalFence.create_at(sem, 1).wait()
        # Map and reformat buffer as np.array.
        raw_dtype = self._get_raw_dtype()
        mapped_memory = host_buffer.map()
        host_array = mapped_memory.asarray(self._buffer_view.shape, raw_dtype)
        # Detect if we need to force an explicit conversion. This happens when
        # we were requested to pretend that the array is in a specific dtype,
        # even if that is not representable on the device. You guessed it:
        # this is to support bools.
        if self._override_dtype is not None and self._override_dtype != raw_dtype:
            host_array = host_array.astype(self._override_dtype)
        return host_array

    def _get_raw_dtype(self):
        return HalElementType.map_to_dtype(self._buffer_view.element_type)

    @property
    def dtype(self):
        if self._override_dtype:
            return self._override_dtype
        return self._get_raw_dtype()

    @property
    def shape(self):
        return np.shape(self)

    def astype(self, dtype, casting="unsafe", copy=True):
        if self.dtype == dtype and not copy:
            return self
        host_ary = self.to_host()
        return host_ary.astype(dtype, casting=casting, copy=copy)

    def reshape(self, *args):
        # TODO(scotttodd): add a native impl with a new buffer_view of the same data
        # TODO(scotttodd): return DeviceArray instead of host ndarray?
        host_ary = self.to_host()
        return host_ary.reshape(*args)

    def __iter__(self):
        host_ary = self.to_host()
        return host_ary.__iter__()

    def __getitem__(self, index):
        host_ary = self.to_host()
        return host_ary.__getitem__(index)

    def __reduce__(self):
        # Since this is used for making deep copies and pickling, we map
        # separately from any interactive state. We just reduce to the actual
        # host ndarray, which supports the necessary serialization protocols.
        _, host_array = self._map_to_host()
        return _restore_reduced_array, (host_array,)


def _restore_reduced_array(ary):
    return ary


# Function implementations with custom behavior.
@_device_implements(np.shape)
def _(arr: DeviceArray):
    return arr._buffer_view.shape


@_device_implements(np.reshape)
def _(arr: DeviceArray, *args):
    return arr.reshape(*args)


def asdevicearray(
    device: HalDevice,
    a,
    dtype=None,
    *,
    implicit_host_transfer: bool = False,
    memory_type=MemoryType.DEVICE_LOCAL,
    allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
    element_type: Optional[HalElementType] = None,
) -> DeviceArray:
    """Helper to create a DeviceArray from an arbitrary array like.

    This is similar in purpose and usage to np.asarray, except that it takes
    a device as the first argument. This may not be the best mechanism for
    getting a DeviceArray, depending on your use case, but it is reliable
    and simple. This function may make a defensive copy or cause implicit
    transfers to satisfy the request. If this is important to you, then a lower
    level API is likely more appropriate.

    Note that additional flags `memory_type`, `allowed_usage` and `element_type`
    are only hints if creating a new DeviceArray. If `a` is already a DeviceArray,
    they are ignored.
    """
    if isinstance(a, DeviceArray):
        if dtype is None:
            return a
        # Need to do a conversion, which we currently do not support on the
        # device, so transfer back to the host.
        logging.warn(
            "Implicit dtype conversion of a DeviceArray forces a host transfer"
        )
    # First get an ndarray. Needs to be C-contiguous, enforcing it here.
    a = np.asarray(a, dtype=dtype, order="C")
    element_type = map_dtype_to_element_type(a.dtype)
    if element_type is None:
        raise ValueError(f"Could not map dtype {a.dtype} to IREE element type")
    buffer_view = device.allocator.allocate_buffer_copy(
        memory_type=memory_type,
        allowed_usage=allowed_usage,
        device=device,
        buffer=a,
        element_type=element_type,
    )
    return DeviceArray(
        device,
        buffer_view,
        implicit_host_transfer=implicit_host_transfer,
        override_dtype=a.dtype,
    )


# NOTE: Numpy dtypes are not hashable and exist in a hierarchy that should
# be queried via isinstance checks. This should be done as a fallback but
# this is a linear list for quick access to the most common. There may also
# be a better way to do this.
_DTYPE_TO_HAL_ELEMENT_TYPE = (
    (np.float16, HalElementType.FLOAT_16),
    (np.float32, HalElementType.FLOAT_32),
    (np.float64, HalElementType.FLOAT_64),
    (np.float16, HalElementType.FLOAT_16),
    (np.int32, HalElementType.SINT_32),
    (np.int64, HalElementType.SINT_64),
    (np.int16, HalElementType.SINT_16),
    (np.int8, HalElementType.SINT_8),
    (np.uint32, HalElementType.UINT_32),
    (np.uint64, HalElementType.UINT_64),
    (np.uint16, HalElementType.UINT_16),
    (np.uint8, HalElementType.UINT_8),
    (np.bool_, HalElementType.BOOL_8),
    (np.complex64, HalElementType.COMPLEX_64),
    (np.complex128, HalElementType.COMPLEX_128),
)


def map_dtype_to_element_type(dtype) -> Optional[HalElementType]:
    for match_dtype, element_type in _DTYPE_TO_HAL_ELEMENT_TYPE:
        if match_dtype == dtype:
            return element_type
    else:
        return None
