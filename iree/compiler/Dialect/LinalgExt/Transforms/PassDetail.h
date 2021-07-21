#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"  // IWYU pragma: keep

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
