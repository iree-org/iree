// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_MESHTOFLOW_PASSES
#define IREE_DIALECT_FLOW_MESHTOFLOW_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMeshToFlow :
    Pass<"iree-convert-mesh-to-flow", "mlir::ModuleOp"> {
  let summary = "Convert Mesh dialect operations to IREE Flow.";
  let description = [{
    Each mesh corresponds to a default Flow channel with the same group name.
    ```
    mesh.mesh @mesh_1(shape = 2x3)
    ```
    ```
    %channel = flow.channel.default "mesh_1" : !flow.channel
    ```
    If there is onl one mesh in the program than the name is omitted and the
    ```
    %channel = flow.channel.default : !flow.channel
    ```

    Each (mesh, mesh_axes) pair partitions and orders the devices into disjoint
    groups, each corresponding to a Flow channel to perform a collective
    operation.
    For example
    ```
    mesh.mesh @mesh(shape = 2x3x4x5)
    ...
    %1 = mesh.all_reduce on @mesh mesh_axes = [2, 0] : tensor<10x20xf32>
    ```
    For more information see
    [Mesh dialect](https://mlir.llvm.org/docs/Dialects/Mesh/#device-groups).

    The mesh partition and device ordering determines the values for the
    `color` and `key` in the corresponding `flow.channel.split` operation used
    to create the channel.
    For more information on the meaning of `color` and `key` see
    [MPI_Comm_split](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node188.htm#Node188)
    in the MPI standard.

    Each Flow channel is wrapped in an IREE `util.global` and its construction
    is done only once with `util.initializer`.
  }];
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertMeshToFlowPass()";
}

#endif  // IREE_DIALECT_FLOW_MESHTOFLOW_PASSES
