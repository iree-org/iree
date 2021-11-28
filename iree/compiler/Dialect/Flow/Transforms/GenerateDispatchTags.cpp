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
} // namespace

class GenerateDispatchTagsPass
    : public GenerateDispatchTagsBase<GenerateDispatchTagsPass> {
 public:
  GenerateDispatchTagsPass() = default;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    // Generate dispatch tag contents and store them as executableop attributes.
    for (auto execOp : getOperation().getOps<IREE::Flow::ExecutableOp>()) {
      // Generate executable IR summary string. 
      auto IRSummary = getIRString(execOp);
      // Store source location tag strings as attributes.
      OpBuilder builder(execOp);;

      SmallVector<NamedAttribute> source_loc_tags = {
      	  builder.getNamedAttr("sourceIRSummary",
      			       builder.getStringAttr(IRSummary)),
      };
      execOp.getOperation()->setAttr("source_loc_tags", builder.getDictionaryAttr(source_loc_tags));
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
