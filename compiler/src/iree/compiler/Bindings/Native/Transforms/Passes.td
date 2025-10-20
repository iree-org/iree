// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_NATIVE_TRANSFORMS_PASSES
#define IREE_BINDINGS_NATIVE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IREE Native ABI bindings support
//===----------------------------------------------------------------------===//

def ConvertStreamableOpsPass :
    Pass<"iree-abi-convert-streamable-ops", "mlir::ModuleOp"> {
  let summary = "Converts streamable ops in input dialects into their IREE dialect forms.";
  let description = [{
    Converts streamable operations from input dialects (like `tensor` and other
    upstream dialects) into their equivalent IREE dialect representations. This
    transformation prepares operations for use with the IREE native ABI
    bindings.
  }];
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Util::UtilDialect",
  ];
}

def WrapEntryPointsPass :
    Pass<"iree-abi-wrap-entry-points", "mlir::ModuleOp"> {
  let summary = "Wraps all entry points in functions compatible with the native IREE ABI.";
  let description = [{
    Wraps all exported and imported functions in wrapper functions that are
    compatible with the expected invocation semantics of bindings following the
    native IREE ABI.

    The wrapper functions handle:
    * Marshaling arguments and results between ABI types and internal representations
    * Managing execution models (sync vs fenced)
    * Preserving internal function types while exposing ABI-compatible interfaces

    The original functions are made private and renamed with a leading
    underscore.
  }];
  let options = [
    Option<"invocationModel", "invocation-model",
           "IREE::ABI::InvocationModel",
           /*default=*/"IREE::ABI::InvocationModel::Sync",
           "Specifies the execution model used for invocations.",
           [{::llvm::cl::values(
               clEnumValN(IREE::ABI::InvocationModel::Sync,
                          "sync",
                          "Fully synchronous behavior with no fences."),
               clEnumValN(IREE::ABI::InvocationModel::CoarseFences,
                          "coarse-fences",
                          "Exposes one wait fence for all inputs and one signal "
                          "fence for all outputs."))}]>,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

#endif // IREE_BINDINGS_NATIVE_TRANSFORMS_PASSES
