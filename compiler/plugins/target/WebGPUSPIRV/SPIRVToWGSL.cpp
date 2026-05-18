// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/WebGPUSPIRV/SPIRVToWGSL.h"

#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "src/tint/lang/core/ir/disassembler.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/program/program.h"
#include "src/tint/lang/wgsl/writer/writer.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

constexpr uint32_t kSPIRVMagicNumber = 0x07230203u;

void printTintFailure(llvm::StringRef stage, llvm::StringRef reason) {
  llvm::errs() << "Tint failed to " << stage;
  if (reason.empty()) {
    llvm::errs() << ".\n";
    return;
  }
  llvm::errs() << ":\n" << reason;
  if (!reason.ends_with('\n')) {
    llvm::errs() << "\n";
  }
}

void printSPIRVModuleSummary(llvm::ArrayRef<uint32_t> spvBinary) {
  llvm::errs() << "SPIR-V module summary:\n";
  llvm::errs() << "  words: " << spvBinary.size() << " ("
               << spvBinary.size() * sizeof(uint32_t) << " bytes)\n";
  if (spvBinary.size() < 5) {
    llvm::errs() << "  header: incomplete; expected at least 5 words\n";
    return;
  }

  uint32_t magic = spvBinary[0];
  uint32_t version = spvBinary[1];
  llvm::errs() << "  magic: " << llvm::format_hex(magic, 10);
  if (magic != kSPIRVMagicNumber) {
    llvm::errs() << " (expected " << llvm::format_hex(kSPIRVMagicNumber, 10)
                 << ")";
  }
  llvm::errs() << "\n";
  llvm::errs() << "  version: " << ((version >> 16) & 0xFF) << "."
               << ((version >> 8) & 0xFF) << " ("
               << llvm::format_hex(version, 10) << ")\n";
  llvm::errs() << "  generator: " << llvm::format_hex(spvBinary[2], 10) << "\n";
  llvm::errs() << "  bound: " << spvBinary[3] << "\n";
  llvm::errs() << "  schema: " << llvm::format_hex(spvBinary[4], 10) << "\n";
}

void printTintIRAtFailure(const tint::core::ir::Module &irModule) {
  llvm::errs() << "Tint IR at failure:\n";
  llvm::errs() << tint::core::ir::Disassembler(irModule).Plain();
  llvm::errs() << "\n";
}

} // namespace

std::optional<std::string>
compileSPIRVToWGSL(llvm::ArrayRef<uint32_t> spvBinary) {
  std::vector<uint32_t> binaryVector(spvBinary.begin(), spvBinary.end());
  auto irResult = tint::spirv::reader::ReadIR(binaryVector);
  if (irResult != tint::Success) {
    printTintFailure("parse SPIR-V into IR", irResult.Failure().reason);
    printSPIRVModuleSummary(spvBinary);
    return std::nullopt;
  }

  tint::core::ir::Module irModule = irResult.Move();
  tint::wgsl::writer::Options writerOptions;
  auto programResult =
      tint::wgsl::writer::ProgramFromIR(irModule, writerOptions);
  if (programResult != tint::Success) {
    printTintFailure("lower Tint IR to a WGSL program",
                     programResult.Failure().reason);
    printSPIRVModuleSummary(spvBinary);
    printTintIRAtFailure(irModule);
    return std::nullopt;
  }

  auto wgslResult =
      tint::wgsl::writer::Generate(programResult.Get(), writerOptions);
  if (wgslResult != tint::Success) {
    printTintFailure("print WGSL", wgslResult.Failure().reason);
    printSPIRVModuleSummary(spvBinary);
    printTintIRAtFailure(irModule);
    return std::nullopt;
  }

  return std::move(wgslResult->wgsl);
}

} // namespace mlir::iree_compiler::IREE::HAL
