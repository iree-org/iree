"""Tracing support."""

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Optional

import logging
import os
import sys

from . import binding as _binding

TRACE_PATH_ENV_KEY = "IREE_SAVE_CALLS"


class Tracer:
  """Object for tracing calls made into the runtime."""

  def __init__(self, trace_path: str):
    self.trace_path = trace_path
    os.makedirs(trace_path, exist_ok=True)
    self._name_count = dict()  # type: Dict[str, int]

  def persist_vm_module(self, vm_module: _binding.VmModule) -> "TracedModule":
    # Depending on how the module was created, there are different bits
    # of information available to reconstruct.
    name = vm_module.name
    flatbuffer_blob = vm_module.stashed_flatbuffer_blob
    if flatbuffer_blob:
      save_path = os.path.join(self.trace_path,
                               self._get_unique_path(f"{name}.vmfb"))
      logging.info("Saving traced vmfb to %s", save_path)
      with open(save_path, "wb") as f:
        f.write(flatbuffer_blob)
      return TracedModule(self, vm_module, save_path)

    # No persistent form, but likely they are built-in modules.
    return TracedModule(self, vm_module)

  def create_invocation(self) -> "InvocationTracer":
    return InvocationTracer(self)

  def _get_unique_path(self, local_name: str) -> str:
    if local_name not in self._name_count:
      self._name_count[local_name] = 1
      return local_name
    stem, ext = os.path.splitext(local_name)
    index = self._name_count[local_name]
    self._name_count[local_name] += 1
    unique_name = f"{stem}__{index}{ext}"
    return unique_name


class TracedModule:

  def __init__(self,
               parent: Tracer,
               vm_module: _binding.VmModule,
               vmfb_path: Optional[str] = None):
    self._parent = parent
    self._vm_module = vm_module
    self._vmfb_path = vmfb_path

  def serialize(self):
    module_record = {"name": self._vm_module.name}
    if self._vmfb_path:
      module_record["vmfb_path"] = os.path.relpath(self._vmfb_path,
                                                   self._parent.trace_path)
    return module_record


class InvocationTracer:

  def __init__(self, parent: Tracer):
    self._parent = parent
    self._modules = []  # type: List[TracedModule]
    self._call_count = 0

  def add_module(self, module: TracedModule):
    self._modules.append(module)

  def start_call(self, function: _binding.VmFunction):
    logging.info("Tracing call to %s.%s", function.module_name, function.name)

    # Start assembling the call record.
    record = {
        "modules": [m.serialize() for m in self._modules],
        "module_name": function.module_name,
        "function_name": function.name,
    }
    return CallTrace(self, record)


class CallTrace:

  def __init__(self, parent: InvocationTracer, record: dict):
    self._parent = parent
    self._record = record

  def end_call(self):
    logging.info("Call ended: %s", repr(self._record))


def get_default_tracer() -> Optional[Tracer]:
  """Gets a default call tracer based on environment variables."""
  default_path = os.getenv(TRACE_PATH_ENV_KEY)
  if not default_path:
    return None
  return Tracer(default_path)
