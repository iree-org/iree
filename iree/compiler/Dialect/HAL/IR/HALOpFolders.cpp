// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

namespace {

/// Simplifies a hal.allocator.compute_size + hal.allocator.allocate pair into
/// a single hal.allocator.allocate_shaped when there are no other paired
/// allocations.
struct SimplifyAllocatorAllocateShapedOp
    : public OpRewritePattern<AllocatorAllocateOp> {
  using OpRewritePattern<AllocatorAllocateOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AllocatorAllocateOp op,
                                     PatternRewriter &rewriter) const override {
    if (auto computeSizeOp = dyn_cast_or_null<AllocatorComputeSizeOp>(
            op.allocation_size()->getDefiningOp())) {
      if (op.memory_types() == computeSizeOp.memory_types() &&
          op.buffer_usage() == computeSizeOp.buffer_usage()) {
        rewriter.replaceOpWithNewOp<AllocatorAllocateShapedOp>(
            op, op.allocator(), op.memory_types(), op.buffer_usage(),
            llvm::to_vector<4>(computeSizeOp.shape()),
            computeSizeOp.element_size().getZExtValue());
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

}  // namespace

void AllocatorAllocateOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAllocatorAllocateShapedOp>(context);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
