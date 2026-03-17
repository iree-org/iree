// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/WebGPUSPIRV/SPIRVToWGSL.h"

#include "llvm/Support/raw_ostream.h"
#include "src/tint/api/tint.h"

namespace mlir::iree_compiler::IREE::HAL {

std::optional<std::string>
compileSPIRVToWGSL(llvm::ArrayRef<uint32_t> spvBinary) {
  std::vector<uint32_t> binaryVector(spvBinary.begin(), spvBinary.end());
  auto result = tint::SpirvToWgsl(binaryVector);
  if (result != tint::Success) {
    llvm::errs() << "Tint SPIR-V to WGSL failed: " << result.Failure().reason
                 << "\n";
    return std::nullopt;
  }
  return result.Move();
}

} // namespace mlir::iree_compiler::IREE::HAL
