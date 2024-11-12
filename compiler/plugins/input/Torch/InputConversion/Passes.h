// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES_H_
#define IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::TorchInput {

// The following is a hard-coded list of ops we don't want to decompose in the
// torch dialect, since they have disadvantageous decompositons for the
// torch-to-linalg path. For example, decomposing `aten.flatten.using_ints` to
// `aten.view` simply destroys useful information about what kind of reshape is
// being performed, and hinders our ability, in some cases, to lower this to a
// collapse instead of a generic reshape.
struct BackendLegalOps {
  static const llvm::SmallVector<std::string> get() {
    return {"aten.flatten.using_ints", "aten.unflatten.int",
            "aten.adaptive_avg_pool1d", "aten.adaptive_avg_pool2d",
            "aten.adaptive_max_pool1d"};
  };
};

struct TorchToIREELoweringPipelineOptions
    : public PassPipelineOptions<TorchToIREELoweringPipelineOptions> {
  Option<bool> strictSymbolicShapes{
      *this, "strict-symbolic-shapes",
      llvm::cl::desc("Use strict symbolic shapes."), llvm::cl::init(true)};
  Option<bool> decompose{*this, "decompose",
                         llvm::cl::desc("Decompose complex torch operations."),
                         llvm::cl::init(true)};
};

// Creates a pipeline that lowers from the torch backend contract to IREE.
// This is based on the torch-backend-to-linalg-on-tensors-backend-pipeline
// pipeline in torch-mlir but includes IREE specific lowerings.
void createTorchToIREEPipeline(
    OpPassManager &pm, const TorchToIREELoweringPipelineOptions &options);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc" // IWYU pragma: keep

void registerTMTensorConversionPasses();

} // namespace mlir::iree_compiler::TorchInput

#endif // IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES_H_
