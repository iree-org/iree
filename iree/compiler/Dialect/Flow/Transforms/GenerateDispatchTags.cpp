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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {
namespace {

//Generate IR at executable scope.
static std::string getIRString(Operation *op) {
  std::string IR;
  llvm::raw_string_ostream out(IR);
  // Print at module scope.
  out << "  ('" << op->getName() << "' operation";
  if (auto symbolName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
    out << ": @" << symbolName.getValue();
  }
  out << ") //----- //\n";

  // Find the top-level operation.
  auto *topLevelOp = op;
  while (auto *parentOp = topLevelOp->getParentOp()) {
    topLevelOp = parentOp;
    out << "  ('" << topLevelOp->getName() << "' operation";
    if (auto symbolName =
            topLevelOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      out << ": @" << symbolName.getValue();
    }
    out << ") //----- //\n";
  }

  return out.str();
}
//Create mlir::Attribute for source code.
static mlir::Attribute getStringAsAttr(std::string str) {
  const void * pointer = str.c_str();
  return Attribute::getFromOpaquePointer(pointer);
}

} // namespace

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
      auto sourcecodeattr = getStringAsAttr(getIRString(funcOp.getOperation()));
      funcOp.getOperation()->setAttr("source_code", sourcecodeattr);
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createGenerateDispatchTagsPass() {
  return std::make_unique<GenerateDispatchTagsPass>();
}


} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
