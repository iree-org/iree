# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Callable, Tuple, Dict, Optional, Any
from .types import Tensor
import iree.runtime
from iree.runtime import VmModule, HalDevice, load_vm_module
from .execution import execute_operation_list
from tempfile import TemporaryDirectory
from iree.compiler.tools import compile_file
import os
from pathlib import Path
from collections import namedtuple
from ._split_mlir import extract_operation_list
from numbers import Number

VmfbFilePath = str
MlirFilePath = str
FunctionName = str


class IreeExecutor:
  """Executor for IREE that implements the `.types.ExecuteOp` interface."""

  def __init__(self, device: HalDevice,
               resolve_function: Callable[[FunctionName], Tuple[VmfbFilePath,
                                                                FunctionName]]):
    """
    Parameters
    ----------
    resolve_function : Resolves a function name that is called in the entry MLIR to
      the vmfb file where it can be found under another name.
    """
    self.device = device
    self.resolve_function = resolve_function

  def __call__(self, op_id: str, operands: List[Tensor]) -> List[Tensor]:
    if op_id.startswith("call "):
      function_name = op_id.split(" ", 2)[1]
      vmfb_file_path, vmfb_function_name = self.resolve_function(function_name)
      config = iree.runtime.Config(device=self.device)
      with open(vmfb_file_path, "rb") as f:
        vm_flatbuffer = f.read()
      vm_module_fb_bytes = VmModule.from_flatbuffer(config.vm_instance,
                                                    vm_flatbuffer)
      vm_module = load_vm_module(vm_module_fb_bytes, config)
      res = getattr(vm_module, vmfb_function_name)(*operands)
      if isinstance(res, (iree.runtime.DeviceArray, Number)):
        res = [res]
      return res
    if op_id == "return":
      return operands
    raise RuntimeError(f"Invalid op_id \"{op_id}\".")


def mlir_to_vmfb_file_path(mlir_file_path: str) -> str:
  return f"{Path(mlir_file_path).stem}.vmfb"


def execute_mlir_with_iree(input: List[Tensor],
                           mlir_path_function_pairs: List[Tuple[MlirFilePath,
                                                                FunctionName]],
                           compile_kwargs: Dict[str, Any],
                           device: HalDevice,
                           override_results: Optional[List[
                               List[Tensor]]] = None,
                           artifact_dir: Optional[str] = None) -> List[Tensor]:
  """Executes an MLIR program that is split accorss multiple MLIR files.
  Parameters
  ----------
  mlir_path_function_pairs : List of MLIR files and the function they contain.
    The first element is the entry MLIR and function.
    It is expected that a name of function called in the entry function correspnd
    to an MLIR file with the same name without file name extension.
  compile_kwargs : Compile arguments to pass to iree.compiler.tools.compile_file.
  artifact_dir : Where to put temporary files.
    Defaults to creating a unique temporary directory that is deleted on completion.

  See: `execute_operation_list`
  """
  if artifact_dir is None:
    with TemporaryDirectory() as temp_dir:
      return execute_mlir_with_iree(
          input=input,
          mlir_path_function_pairs=mlir_path_function_pairs,
          override_results=override_results,
          compile_kwargs=compile_kwargs,
          device=device,
          artifact_dir=temp_dir)

  entry_mlir_file_path = mlir_path_function_pairs[0][0]
  entry_function_name = mlir_path_function_pairs[0][1]
  FunctionDescription = namedtuple(
      "FunctionDescription",
      ["mlir_file_path", "vmfb_file_path", "function_name"])
  function_map = {
      Path(Path(p[0]).name).stem: FunctionDescription(
          p[0], os.path.join(artifact_dir, mlir_to_vmfb_file_path(p[0])), p[1])
      for p in mlir_path_function_pairs
  }
  for i in range(1, len(mlir_path_function_pairs)):
    function_description = function_map[Path(
        Path(mlir_path_function_pairs[i][0]).name).stem]
    compile_file(function_description.mlir_file_path,
                 output_file=function_description.vmfb_file_path,
                 **compile_kwargs)

  def resolve_function(
      function_name: FunctionName) -> Tuple[VmfbFilePath, FunctionName]:
    func_desc = function_map[function_name]
    return (func_desc.vmfb_file_path, func_desc.function_name)

  executor = IreeExecutor(device=device, resolve_function=resolve_function)
  operation_list = extract_operation_list(mlir_file_path=entry_mlir_file_path,
                                          function_name=entry_function_name)
  return execute_operation_list(operation_list=operation_list,
                                execute_op=executor,
                                input=input,
                                override_results=override_results)
