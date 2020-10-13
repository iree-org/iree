
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

llvm::SmallVector<int64_t, 4> getTileSizesImpl(linalg::MatmulOp op) {
  return {128, 128};
}

llvm::SmallVector<int64_t, 4> CPUKernelDispatch::getTileSizes(
    Operation* op) const {
  if (isa<linalg::MatmulOp>(op)) {
    return getTileSizesImpl(dyn_cast<linalg::MatmulOp>(op));
  }
  return {1, 1, 1};
}

}  // namespace iree_compiler
}  // namespace mlir
