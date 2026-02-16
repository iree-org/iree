#include "Gemmini/GemminiOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::buddy {
namespace {

class GemminiIRDumpsPass
    : public PassWrapper<GemminiIRDumpsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiIRDumpsPass)

  GemminiIRDumpsPass() = default;
  GemminiIRDumpsPass(const GemminiIRDumpsPass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "iree-codegen-gemmini-ir-dump"; }
  StringRef getDescription() const final {
    return "Dumps IR for Gemmini matmul/batch_matmul/conv patterns";
  }

  Option<bool> dumpLinalg{
      *this, "dump-linalg",
      llvm::cl::desc("Dump Linalg matmul/batch_matmul/conv ops"),
      llvm::cl::init(true)};
  Option<bool> dumpGemmini{
      *this, "dump-gemmini",
      llvm::cl::desc("Dump Gemmini tile_matmul/tile_conv ops"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &os = llvm::errs();

    module.walk([&](Operation *op) {
      bool matched = false;
      if (dumpLinalg &&
          isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::Conv2DNchwFchwOp,
              linalg::Conv2DNhwcHwcfOp>(op)) {
        matched = true;
      }
      if (dumpGemmini &&
          isa<::buddy::gemmini::TileMatMulOp, ::buddy::gemmini::TileConvOp>(op)) {
        matched = true;
      }
      if (!matched)
        return;

      os << "GEMMINI_IR_DUMP: " << op->getName() << "\n";
      if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
        os << "GEMMINI_IR_DUMP_FUNC: @" << funcOp.getName() << "\n";
      }
      op->print(os, OpPrintingFlags().useLocalScope());
      os << "\n";
    });
  }
};

} // namespace

void registerGemminiIRDumpsPass() { PassRegistration<GemminiIRDumpsPass>(); }

} // namespace mlir::buddy
