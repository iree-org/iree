// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_ANALYSIS_RESOURCE_HAZARDS_H_
#define IREE_COMPILER_DIALECT_STREAM_ANALYSIS_RESOURCE_HAZARDS_H_

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Hazard analysis
//===----------------------------------------------------------------------===//

// Performs localized analysis of resource hazards.
// All ops producing and consuming `!stream.resource` SSA values will be
// analyzed and the hazards between them will be available for querying via the
// lookup functions.
class ResourceHazardAnalysis {
public:
  explicit ResourceHazardAnalysis(Operation *rootOp);
  ~ResourceHazardAnalysis();

  // Runs analysis and populates the hazard map.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run();

  // Returns true if there is a hazard between |producerOp| and |consumerOp|.
  // A hazard indicates that the consumer must wait for the producer to complete
  // execution prior to beginning execution. By default any resource SSA value
  // will induce a hazard but certain ops acting on exclusive subranges may be
  // allowed to run while operating on the same resource.
  bool hasHazard(Operation *producerOp, Operation *consumerOp);

private:
  std::unique_ptr<AsmState> asmState;
};

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_ANALYSIS_RESOURCE_HAZARDS_H_
