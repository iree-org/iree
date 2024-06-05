// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_GENERATESPLATPARAMETERARCHIVEPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

namespace {

// Returns a set of all unique parameters and the locations using them.
static SmallVector<std::pair<Location, IREE::Flow::NamedParameterAttr>>
findAllParameters(ModuleOp moduleOp) {
  llvm::MapVector<IREE::Flow::NamedParameterAttr, SmallVector<Location>>
      parameterAttrs;
  moduleOp.walk([&](Operation *op) {
    if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
      if (auto parameterAttr =
              dyn_cast_if_present<IREE::Flow::NamedParameterAttr>(
                  globalOp.getGlobalInitialValue())) {
        parameterAttrs[parameterAttr].push_back(globalOp.getLoc());
      }
    } else if (auto constantOp = dyn_cast<IREE::Flow::TensorConstantOp>(op)) {
      if (auto parameterAttr =
              dyn_cast_if_present<IREE::Flow::NamedParameterAttr>(
                  constantOp.getValue())) {
        parameterAttrs[parameterAttr].push_back(constantOp.getLoc());
      }
    }
  });
  SmallVector<std::pair<Location, IREE::Flow::NamedParameterAttr>> locAttrs;
  for (auto &entry : parameterAttrs) {
    locAttrs.push_back(std::make_pair(
        FusedLoc::get(moduleOp.getContext(), entry.second), entry.first));
  }
  return locAttrs;
}

static Attribute getDefaultSplatAttr(Type elementType) {
  // Today we only support basic types where 0 bits represent zeros - that lets
  // us just splat out the right number of bits.
  int64_t storageSize =
      IREE::Util::getRoundedPhysicalStorageSize(1, elementType);
  return IntegerAttr::get(
      IntegerType::get(elementType.getContext(), storageSize * 8), 0);
}

struct GenerateSplatParameterArchivePass
    : public IREE::IO::Parameters::impl::GenerateSplatParameterArchivePassBase<
          GenerateSplatParameterArchivePass> {
  using IREE::IO::Parameters::impl::GenerateSplatParameterArchivePassBase<
      GenerateSplatParameterArchivePass>::GenerateSplatParameterArchivePassBase;

  void runOnOperation() override {
    // Nothing to do if no path specified.
    if (filePath.empty())
      return;

    // Create a builder used to accumulate the parameters.
    ModuleOp moduleOp = getOperation();
    auto builder = createArchiveBuilder(moduleOp);
    if (failed(builder))
      return signalPassFailure();

    // Find all parameters in the module and add them to the builder.
    // NOTE: there may be no parameters but we still will create the archive
    // so that subsequent tooling that tries to load it succeeds.
    auto parameterAttrs = findAllParameters(moduleOp);
    for (auto [loc, parameterAttr] : parameterAttrs) {
      // Only support types we can meaningfully generate splats for.
      auto shapedType = dyn_cast<ShapedType>(parameterAttr.getType());
      if (!shapedType)
        continue;

      // TODO: support other patterns/generators.
      auto elementAttr = getDefaultSplatAttr(shapedType.getElementType());

      // Serialize the splat pattern.
      SmallVector<char, IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH> pattern;
      llvm::raw_svector_ostream os(pattern);
      if (failed(IREE::Util::SerializableAttrInterface::serializeSplatValue(
              loc, elementAttr,
              /*count=*/1, llvm::endianness::little, os))) {
        return signalPassFailure();
      }

      // Add a zero-splat entry to the builder for this global.
      auto parameterName = parameterAttr.getKey().getValue();
      if (failed(handleRuntimeError(
              moduleOp,
              iree_io_parameter_archive_builder_add_splat_entry(
                  builder->get(),
                  iree_string_view_t{
                      parameterName.data(),
                      static_cast<iree_host_size_t>(parameterName.size())},
                  /*metadata=*/iree_const_byte_span_empty(),
                  /*pattern=*/pattern.data(),
                  /*pattern_length=*/static_cast<uint8_t>(pattern.size()),
                  /*data_length=*/parameterAttr.getStorageSize()),
              "failed to add splat entry for global"))) {
        return signalPassFailure();
      }
    }

    // Create the parameter archive file.
    auto fileStreamIndexOr =
        createParameterIndex(moduleOp, std::move(builder.value()), filePath);
    if (failed(fileStreamIndexOr))
      return signalPassFailure();
    auto [file, stream, index] = *std::move(fileStreamIndexOr);

    // Commit the written file.
    if (llvm::Error maybeCommit = file->commit()) {
      InFlightDiagnostic errorStream =
          moduleOp.emitError() << "Failed to commit archive with error: ";
      llvm::handleAllErrors(std::move(maybeCommit),
                            [&](const llvm::ErrorInfoBase &PE) {
                              errorStream << PE.message() << "\n";
                            });
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::IO::Parameters
