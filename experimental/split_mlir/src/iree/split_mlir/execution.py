# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Optional
from .types import OpArguments, Tensor, ExecuteOp, OperationList


def collect_arguments(arguments: OpArguments,
                      results: List[List[Tensor]]) -> List[Tensor]:
  return [results[arg[0]][arg[1]] for arg in arguments]  # type: ignore


def execute_operation_list(
    input: List[Tensor],
    operation_list: OperationList,
    execute_op: ExecuteOp,
    override_results: Optional[List[List[Tensor]]] = None
) -> List[List[Tensor]]:
  """Algorithm to execute a call list.

  Parameters
  ----------
  input : Input of the graph.
  execute_op : Callable that executes an operation from the graph.
  override_results : When execting operations override arguments with this values,
    instead of using the computed resuts from previous functions.
  
  Returns
  -------
  All results from all operations is the graph are in the same order
  as they appear in `operation_list`. `input` is prepened to the result.
  """
  results = [input]
  for op in operation_list[1:]:
    arguments = collect_arguments(
        arguments=op[1],
        results=results if override_results is None else override_results)
    results.append(execute_op(op[0], arguments))
  return results
