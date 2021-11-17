#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h" 
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h" 
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {
namespace {

//Generate IR at executable scope.
static std::string executableIRString(Operation *op) {
  auto function = dyn_cast<FuncOp>(op);
  std::string execIR;
  llvm::raw_string_ostream out(execIR)
  // Print the function name and a newline before the Module.
  out << " (function: " << function.getName() << ")\n";
  function.getParentOfType<ModuleOp>().print(out);

  // Print a newline before the IR.
  out << "\n";

  // Print the given function.
  if (function) {
    function.print(out);
    return;
  }

  // Print the given module.
  assert(isa<ModuleOp>(op) && "unexpected IR unit");
  cast<ModuleOp>(op).print(out);
  return out.str();
}

class GenerateDispatchTagsPass
    : public GenerateDispatchTagsBase<GenerateDispatchTagsPass> {
 public:
  GenerateDispatchTagsPass() = default;
 
  void runOnOperation() override {
    // Generate module IR string for each dispatch region and store
    // it as a dispatch attribute..
    for (auto funcOp : getOperation().getOps<mlir::FuncOp>()) {
      // Generate module IR string. 
      // Store IR string as attribute.
      funcOp.setAttr("source_code", executableIRString(*funcOp);
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createGenerateDispatchTagsPass() {
  return std::make_unique<GenerateDispatchTagsPass>();
}

} // namespace
} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
