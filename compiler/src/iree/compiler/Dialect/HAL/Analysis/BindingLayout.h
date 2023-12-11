// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_ANALYSIS_BINDINGLAYOUT_
#define IREE_COMPILER_DIALECT_HAL_ANALYSIS_BINDINGLAYOUT_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

struct DescriptorSetLayoutBinding {
  // Ordinal of the descriptor within its parent set layout.
  unsigned ordinal;
  // Storage type of the descriptor resource.
  IREE::HAL::DescriptorType type;
  // Flags defining how the descriptor behaves.
  IREE::HAL::DescriptorFlags flags;
};

struct DescriptorSetLayout {
  // Ordinal of the set within the parent pipeline layout.
  unsigned ordinal;
  // Usage of the descriptor set (such as whether it is persistent or push).
  IREE::HAL::DescriptorSetLayoutFlags flags;
  // Bindings within the layout. Ordinals may be sparse.
  SmallVector<DescriptorSetLayoutBinding> bindings;
};

using PipelineResourceMap = SmallVector<std::pair<unsigned, unsigned>>;

struct PipelineLayout {
  // Total number of 32-bit push constants allocated. Not all dispatchable
  // functions using this layout will use all constants.
  int64_t pushConstantCount;
  // Sets bound in the layout. Ordinals may be sparse.
  SmallVector<DescriptorSetLayout> setLayouts;
  // Mapping of flattened source resource bindings into the descriptor sets.
  // Matches 1:1 with the IREE::Stream::CmdDispatchOp::resources.
  PipelineResourceMap resourceMap;

  void print(llvm::raw_ostream &os) const;
};

// Analyzes dispatch ops and plans out the descriptor set layouts used during
// execution of device code. This attempts to reduce the total number of
// descriptor sets that will be required and frequency at which they are
// updated.
//
// NOTE: erasing dispatch ops will invalidate this analysis.
class BindingLayoutAnalysis {
public:
  using ExportDispatchMap =
      DenseMap<Operation *, SmallVector<IREE::Stream::CmdDispatchOp>>;
  using ExportLayoutMap = DenseMap<Operation *, PipelineLayout>;

  explicit BindingLayoutAnalysis(Operation *rootOp);

  // Returns all of the dispatches to the given executable export.
  SmallVector<IREE::Stream::CmdDispatchOp>
  getExportDispatches(IREE::Stream::ExecutableExportOp exportOp) const;

  // Returns a layout used for the given executable export op.
  const PipelineLayout &
  getPipelineLayout(IREE::Stream::ExecutableExportOp exportOp) const;

private:
  // All dispatches to a particular executable IREE::Stream::ExecutableExportOp.
  ExportDispatchMap exportDispatches;
  // Pipeline layout for each IREE::Stream::ExecutableExportOp.
  // Many of these may be duplicates.
  ExportLayoutMap exportLayouts;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_ANALYSIS_BINDINGLAYOUT_
