# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import ctypes
import os

__all__ = [
    "Invocation",
    "Session",
]

_dylib = None

_GET_FLAG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_void_p)


def _setsig(f, restype, argtypes):
  f.restype = restype
  f.argtypes = argtypes


def _init_dylib():
  global _dylib
  if _dylib:
    return
  dylib_path = os.getenv("OPENXLA_PARTITIONER_LIB_PATH")
  if dylib_path is None:
    # TODO: Look for a bundled dylib.
    raise RuntimeError("Could not find libOpenXLAPartitioner.so: "
                       "set OPENXLA_PARTITIONER_LIB_PATH")
  _dylib = ctypes.cdll.LoadLibrary(dylib_path)

  # Setup signatures.
  _setsig(_dylib.openxlaPartitionerErrorDestroy, None, [ctypes.c_void_p])
  _setsig(_dylib.openxlaPartitionerErrorGetMessage, ctypes.c_char_p,
          [ctypes.c_void_p])
  _setsig(_dylib.openxlaPartitionerInvocationCreate, ctypes.c_void_p,
          [ctypes.c_void_p])
  _setsig(_dylib.openxlaPartitionerInvocationDestroy, None, [ctypes.c_void_p])
  _setsig(_dylib.openxlaPartitionerSessionCreate, ctypes.c_void_p, [])
  _setsig(_dylib.openxlaPartitionerSessionDestroy, None, [ctypes.c_void_p])
  _setsig(_dylib.openxlaPartitionerSessionGetFlags, None,
          [ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p])
  _setsig(_dylib.openxlaPartitionerSessionSetFlags, ctypes.c_void_p,
          [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p])


def _handle_error(err_p, exc_type=ValueError):
  if err_p is None:
    return
  message = _dylib.openxlaPartitionerErrorGetMessage(err_p).decode("UTF-8")
  _dylib.openxlaPartitionerErrorDestroy(err_p)
  raise exc_type(message)


def _global_initialize():
  _dylib.openxlaPartitionerGlobalInitialize()


def _global_shutdown():
  _dylib.openxlaPartitionerGlobalShutdown()


class _GlobalInit:

  def __init__(self):
    _init_dylib()
    _dylib.openxlaPartitionerGlobalInitialize()

  def __del__(self):
    _dylib.openxlaPartitionerGlobalShutdown()


# Keep one reference for the life of the module.
_global_init = _GlobalInit()


class Session:

  def __init__(self):
    self._global_init = _global_init
    self._session_p = _dylib.openxlaPartitionerSessionCreate()

  def __del__(self):
    _dylib.openxlaPartitionerSessionDestroy(self._session_p)

  def invocation(self):
    return Invocation(self)

  def get_flags(self, non_default_only: bool = False) -> Sequence[str]:
    results = []

    @_GET_FLAG_CALLBACK
    def callback(flag_pointer, length, user_data):
      flag_bytes = ctypes.string_at(flag_pointer, length)
      flag_value = flag_bytes.decode("UTF-8")
      results.append(flag_value)

    _dylib.openxlaPartitionerSessionGetFlags(self._session_p, non_default_only,
                                             callback, ctypes.c_void_p(0))
    return results

  def set_flags(self, *flags: Sequence[str]):
    argv_type = ctypes.c_char_p * len(flags)
    argv = argv_type(*[flag.encode("UTF-8") for flag in flags])
    _handle_error(
        _dylib.openxlaPartitionerSessionSetFlags(self._session_p, len(argv),
                                                 argv))


class Invocation:

  def __init__(self, session: Session):
    self._session = session
    self._inv_p = _dylib.openxlaPartitionerInvocationCreate(
        self._session._session_p)

  def __del__(self):
    _dylib.openxlaPartitionerInvocationDestroy(self._inv_p)
