#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

#define GEN_PASS_CLASSES
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h.inc"  // IWYU pragma: keep

}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
