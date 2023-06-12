// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PASSES_H_
#define IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PASSES_H_

#include "iree/compiler/InputConversion/StableHLO/PassDetail.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class TypeConverter;
namespace iree_compiler::stablehlo {

std::unique_ptr<TypeConverter> createStableHloToLinalgTypeConverter();

struct StableHloOptions : public PassPipelineOptions<StableHloOptions> {
  bool demoteI64ToI32 = true;
  bool demoteF64ToF32 = false;
  bool promoteBF16ToF32 = false;
};

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

void buildStableHLOInputConversionPassPipeline(OpPassManager& passManager,
                                               const StableHloOptions& options);

// Performs input legalization on programs that may have originated from an XLA
// import (or made to interop with it).
void buildStableHLOXLAInputConversionPassPipeline(
    OpPassManager& passManager, const StableHloOptions& options);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerStableHLOConversionPasses();

}  // namespace iree_compiler::stablehlo
}  // namespace mlir
#endif  // IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PASSES_H_
