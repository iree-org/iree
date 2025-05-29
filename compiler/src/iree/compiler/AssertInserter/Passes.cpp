#include "iree/compiler/AssertInserter/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
//===---------------------------------------------------------------------===//
// Include pass headers per target device
//===---------------------------------------------------------------------===//
//#include "iree/compiler/Codegen/Common/CPU/Passes.h"
//#include "iree/compiler/Codegen/Common/GPU/Passes.h"
//#include "iree/compiler/Codegen/Common/Passes.h"
//#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
//#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
//#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
//#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
//#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
//#include "iree/compiler/Codegen/SPIRV/Passes.h"
//#include "iree/compiler/Codegen/VMVX/Passes.h"
//#include "iree/compiler/Codegen/WGSL/Passes.h"

namespace mlir::iree_compiler {

/*
namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/AssertInserter/Passes.h.inc" // IWYU pragma: export
} // namespace
*/

void buildAssertInserterPipeline(mlir::OpPassManager &pm,
                                 const AssertInserterPipelineOptions &options) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::iree_compiler::createAssertInBoundsPass(
      mlir::iree_compiler::AssertInBoundsPassOptions{
          options.warnOnUnknown, options.includeVectorLoadStore,
          options.checkEachDim, options.createSpeculativeFuncs}));
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createCheckStaticAssertionsPass());
}

} // namespace mlir::iree_compiler
