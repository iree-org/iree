#ifndef COMPILER_CODEGEN_LLVMCPU_ENCODINGINFO_H_
#define COMPILER_CODEGEN_LLVMCPU_ENCODINGINFO_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

IREE::LinalgExt::MaterializeEncodingValueFn getMaterializeEncodingValueFn(
    IREE::HAL::ExecutableTargetAttr targetAttr);

void populateLLVMCPUDispatchWorkgroupCountPatterns(RewritePatternSet &patterns,
                                                   Operation *dispatchRootOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // COMPILER_CODEGEN_LLVMCPU_ENCODINGINFO_H_
