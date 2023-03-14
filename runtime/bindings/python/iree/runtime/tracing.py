"""Tracing support."""

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from genericpath import exists
from typing import Dict, List, Optional, Sequence

import logging
import os
import sys

from . import _binding

try:
  import yaml
except ModuleNotFoundError:
  _has_yaml = False
else:
  _has_yaml = True

__all__ = [
    "get_default_tracer",
    "Tracer",
    "TRACE_PATH_ENV_KEY",
]

TRACE_PATH_ENV_KEY = "IREE_SAVE_CALLS"


class Tracer:
  """Object for tracing calls made into the runtime."""

  def __init__(self, trace_path: str):
    if not _has_yaml:
      self.enabled = False
      logging.warning("PyYAML not installed: tracing will be disabled")
      return
    self.enabled = True
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
                               self.get_unique_name(f"{name}.vmfb"))
      logging.info("Saving traced vmfb to %s", save_path)
      with open(save_path, "wb") as f:
        f.write(flatbuffer_blob)
      return TracedModule(self, vm_module, save_path)

    # No persistent form, but likely they are built-in modules.
    return TracedModule(self, vm_module)

  def get_unique_name(self, local_name: str) -> str:
    if local_name not in self._name_count:
      self._name_count[local_name] = 1
      return local_name
    stem, ext = os.path.splitext(local_name)
    index = self._name_count[local_name]
    self._name_count[local_name] += 1
    unique_name = f"{stem}__{index}{ext}"
    return unique_name


class TracedModule:
  """Wraps a VmModule with additional information for tracing."""

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
      module_record["type"] = "bytecode"
      module_record["path"] = os.path.relpath(self._vmfb_path,
                                              self._parent.trace_path)
    else:
      module_record["type"] = "builtin"

    return module_record


class ContextTracer:
  """Traces invocations against a context."""

  def __init__(self, parent: Tracer, is_dynamic: bool,
               modules: Sequence[TracedModule]):
    self._parent = parent
    self._modules = list(modules)  # type: List[TracedModule]
    self._frame_count = 0
    self._file_path = os.path.join(parent.trace_path,
                                   parent.get_unique_name("calls.yaml"))
    if os.path.exists(self._file_path):
      # Truncate the file.
      with open(self._file_path, "wt"):
        pass
    else:
      os.makedirs(os.path.dirname(parent.trace_path), exist_ok=True)
    logging.info("Tracing context events to: %s", self._file_path)
    self.emit_frame({
        "type": "context_load",
    })
    for module in self._modules:
      self.emit_frame({
          "type": "module_load",
          "module": module.serialize(),
      })

  def add_module(self, module: TracedModule):
    self._modules.append(module)
    self.emit_frame({
        "type": "module_load",
        "module": module.serialize(),
    })

  def start_call(self, function: _binding.VmFunction):
    logging.info("Tracing call to %s.%s", function.module_name, function.name)

    # Start assembling the call record.
    record = {
        "type": "call",
        "function": "%s.%s" % (function.module_name, function.name),
    }
    return CallTrace(self, record)

  def emit_frame(self, frame: dict):
    self._frame_count += 1
    with open(self._file_path, "at") as f:
      if self._frame_count != 1:
        f.write("---\n")
      contents = yaml.dump(frame, sort_keys=False)
      f.write(contents)


class CallTrace:

  def __init__(self, parent: ContextTracer, record: dict):
    self._parent = parent
    self._record = record

  def add_vm_list(self, vm_list: _binding.VmVariantList, key: str):
    mapped = []
    for i in range(len(vm_list)):
      mapped.append(vm_list.get_serialized_trace_value(i))
    self._record[key] = mapped

  def end_call(self):
    self._parent.emit_frame(self._record)


def get_default_tracer() -> Optional[Tracer]:
  """Gets a default call tracer based on environment variables."""
  default_path = os.getenv(TRACE_PATH_ENV_KEY)
  if not default_path:
    return None
  return Tracer(default_path)
