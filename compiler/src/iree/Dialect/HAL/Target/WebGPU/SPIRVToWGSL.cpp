// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/WebGPU/SPIRVToWGSL.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "tint/tint.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

llvm::Optional<std::string> compileSPIRVToWGSL(
    llvm::ArrayRef<uint32_t> spvBinary) {
  // TODO(scotttodd): reroute to MLIR diagnostics?
  auto diagPrinter = tint::diag::Printer::create(stderr, true);
  tint::diag::Formatter diagFormatter;

  // TODO(scotttodd): remove this copy (API for std::span or [uint8_t*, size]?)
  std::vector<uint32_t> binaryVector(spvBinary.size());
  std::memcpy(binaryVector.data(), spvBinary.data(),
              spvBinary.size() * sizeof(uint32_t));

  auto program =
      std::make_unique<tint::Program>(tint::reader::spirv::Parse(binaryVector));
  if (!program) {
    llvm::errs() << "Tint failed to parse SPIR-V program\n";
    return llvm::None;
  }

  if (program->Diagnostics().contains_errors()) {
    llvm::errs() << "Tint reported " << program->Diagnostics().error_count()
                 << " error(s) for a SPIR-V program, see diagnostics:\n";
    diagFormatter.format(program->Diagnostics(), diagPrinter.get());
    return llvm::None;
  }

  if (!program->IsValid()) {
    llvm::errs() << "Tint parsed an invalid SPIR-V program\n";
    return llvm::None;
  }

  // TODO(scotttodd): Refine this set of transforms
  tint::transform::Manager transformManager;
  tint::transform::DataMap transformInputs;
  transformInputs.Add<tint::transform::FirstIndexOffset::BindingPoint>(0, 0);
  transformManager.Add<tint::transform::FirstIndexOffset>();
  transformManager.Add<tint::transform::FoldTrivialSingleUseLets>();

  auto output = transformManager.Run(program.get(), std::move(transformInputs));
  if (!output.program.IsValid()) {
    llvm::errs() << "Tint transforms failed on the parsed SPIR-V program\n";
    diagFormatter.format(output.program.Diagnostics(), diagPrinter.get());
    return llvm::None;
  }

  tint::writer::wgsl::Options genOptions;
  auto result = tint::writer::wgsl::Generate(&output.program, genOptions);
  if (!result.success) {
    llvm::errs() << "Tint failed to generate WGSL: " << result.error << "\n";
    return llvm::None;
  }

  return result.wgsl;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
