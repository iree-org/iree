// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MIPS_OPS
#define IREE_DIALECT_MIPS_OPS

include "iree/compiler/Dialect/MIPS/IR/MIPSBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// mips.matmul — tensor-only semantic op
//
// This op exists exclusively in the tensor domain.  It is eliminated during
// One-Shot Bufferize: the BufferizableOpInterface implementation allocates
// the output memref and emits func.call @mips_matmul(...) directly, so no
// memref form of this op ever exists in the IR.
//===----------------------------------------------------------------------===//

def MIPS_MatmulOp : MIPS_Op<"matmul", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
    DestinationStyleOpInterface
]> {
  let summary = "MIPS matrix multiplication (tensor domain): result = lhs * rhs";
  let description = [{
    Computes a 2-D matrix multiplication in the tensor domain using
    destination-passing style (DPS). The caller provides an `init` tensor that
    One-Shot Bufferize uses to determine the output buffer — typically
    `bufferization.alloc_tensor` for a fresh allocation.

    The semantic is: `result[m, n] = sum_k(lhs[m, k] * rhs[k, n])`.

    This op is created by the Torch -> MIPS conversion pass from `torch.aten.mm`
    and is eliminated entirely during bufferization: the BufferizableOpInterface
    implementation emits `func.call @mips_matmul` directly with the bufferized
    memref operands.  No memref-form `mips.matmul` is ever produced.

    Example:
    ```mlir
    %result = mips.matmul %A, %B, %init
        : tensor<4x8xf32>, tensor<8x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,    // [M x K]
    AnyRankedTensor:$rhs,    // [K x N]
    AnyRankedTensor:$init    // [M x N] — DPS destination (typically alloc_tensor)
  );

  let results = (outs AnyRankedTensor:$result);  // [M x N]

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $init attr-dict `:`
    type($lhs) `,` type($rhs) `,` type($init) `->` type($result)
  }];

  let extraClassDeclaration = [{
    // DestinationStyleOpInterface: init is the DPS output operand.
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];

  let hasVerifier = 1;
}

#endif // IREE_DIALECT_MIPS_OPS
