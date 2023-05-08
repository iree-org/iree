// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_ANALYSIS_RESOURCE_USAGE_H_
#define IREE_COMPILER_DIALECT_STREAM_ANALYSIS_RESOURCE_USAGE_H_

#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Resource usage bits
//===----------------------------------------------------------------------===//

enum class ResourceUsageBitfield : uint32_t {
  Indirect = 1u << 0,
  External = 1u << 1,
  Mutated = 1u << 2,  // beyond definition
  Constant = 1u << 3,
  TransferRead = 1u << 4,
  TransferWrite = 1u << 5,
  StagingRead = 1u << 6,
  StagingWrite = 1u << 7,
  DispatchRead = 1u << 8,
  DispatchWrite = 1u << 9,
  GlobalRead = 1u << 10,
  GlobalWrite = 1u << 11,

  Unknown = Indirect | External | Mutated | Constant | TransferRead |
            TransferWrite | StagingRead | StagingWrite | DispatchRead |
            DispatchWrite | GlobalRead | GlobalWrite,
};

inline ResourceUsageBitfield operator|(ResourceUsageBitfield lhs,
                                       ResourceUsageBitfield rhs) {
  return static_cast<ResourceUsageBitfield>(static_cast<uint32_t>(lhs) |
                                            static_cast<uint32_t>(rhs));
}

inline ResourceUsageBitfield operator&(ResourceUsageBitfield lhs,
                                       ResourceUsageBitfield rhs) {
  return static_cast<ResourceUsageBitfield>(static_cast<uint32_t>(lhs) &
                                            static_cast<uint32_t>(rhs));
}

inline bool bitEnumContains(ResourceUsageBitfield bits,
                            ResourceUsageBitfield bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

//===----------------------------------------------------------------------===//
// Resource usage analysis
//===----------------------------------------------------------------------===//

// Performs whole-program analysis of resource usage.
// All `!stream.resource` SSA values will be analyzed and their usage will be
// available for querying via the lookup functions.
class ResourceUsageAnalysis {
 public:
  explicit ResourceUsageAnalysis(Operation *rootOp);
  ~ResourceUsageAnalysis();

  // Runs analysis and populates the resource usage map.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run();

  // Returns the analyzed resource usage of the |value| resource.
  // May return `Unknown` if analysis failed for the given value due to usage
  // that lead to indeterminate results (such as indirect access).
  ResourceUsageBitfield lookupResourceUsage(Value value) {
    return tryLookupResourceUsage(value).value_or(
        ResourceUsageBitfield::Unknown);
  }

  // Returns the analyzed resource usage of the |value| resource, if analyzed.
  std::optional<ResourceUsageBitfield> tryLookupResourceUsage(Value value);

 private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
};

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_STREAM_ANALYSIS_RESOURCE_USAGE_H_
