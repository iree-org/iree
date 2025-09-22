#include <cstdint>

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_GETPADDINGVALUEPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {
class GetPaddingValuePass
    : public impl::GetPaddingValuePassBase<GetPaddingValuePass> {
public:
  void runOnOperation() override;
};
} // namespace

void GetPaddingValuePass::runOnOperation() {
  llvm::errs() << "Starting Exsleratev2 padding analysis pass...\n";

  // The pass should operate on the ModuleOp.
  auto moduleOp = getOperation();
  if (!moduleOp) {
    // This check is a safeguard; getOperation() should not return null for a
    // ModuleOp pass.
    return signalPassFailure();
  }
  OpBuilder rewriter(moduleOp->getContext());

  moduleOp->walk([&](Operation *op) {
    if (auto padOp = dyn_cast<mlir::tensor::PadOp>(op)) {
      auto lowPaddingAttr = padOp.getStaticLow();
      auto highPaddingAttr = padOp.getStaticHigh();

      if (lowPaddingAttr.empty() || highPaddingAttr.empty()) {
        return;
      }

      for (Operation *user : padOp.getResult().getUsers()) {
        if (auto convOp = dyn_cast<linalg::LinalgOp>(user)) {
          if (auto convOp2 = llvm::dyn_cast<linalg::Conv2DNchwFchwOp>(
                  convOp.getOperation())) {
            int64_t padH_low = lowPaddingAttr[2];
            int64_t padH_high = highPaddingAttr[2];
            int64_t padW_low = lowPaddingAttr[3];
            int64_t padW_high = highPaddingAttr[3];

            // llvm::outs() << "Found NCHW convolution with padding:" << "
            // H_low: "
            //              << padH_low << ", H_high: " << padH_high
            //              << ", W_low: " << padW_low
            //              << ", W_high: " << padW_high;

            SmallVector<NamedAttribute> paddingAttrs;
            paddingAttrs.push_back({rewriter.getStringAttr("padH_low"),
                                    rewriter.getI64IntegerAttr(padH_low)});
            paddingAttrs.push_back({rewriter.getStringAttr("padH_high"),
                                    rewriter.getI64IntegerAttr(padH_high)});
            paddingAttrs.push_back({rewriter.getStringAttr("padW_low"),
                                    rewriter.getI64IntegerAttr(padW_low)});
            paddingAttrs.push_back({rewriter.getStringAttr("padW_high"),
                                    rewriter.getI64IntegerAttr(padW_high)});

            convOp2->setAttr("exsleratev2.padding",
                             rewriter.getDictionaryAttr(paddingAttrs));

            return;
          } else if (auto convOp2 = llvm::dyn_cast<linalg::Conv2DNhwcFhwcOp>(
                         convOp.getOperation())) {
            int64_t padH_low = lowPaddingAttr[1];
            int64_t padH_high = highPaddingAttr[1];
            int64_t padW_low = lowPaddingAttr[2];
            int64_t padW_high = highPaddingAttr[2];

            // llvm::outs() << "Found NCHW convolution with padding:" << "
            // H_low: "
            //              << padH_low << ", H_high: " << padH_high
            //              << ", W_low: " << padW_low
            //              << ", W_high: " << padW_high;

            SmallVector<NamedAttribute> paddingAttrs;
            paddingAttrs.push_back({rewriter.getStringAttr("padH_low"),
                                    rewriter.getI64IntegerAttr(padH_low)});
            paddingAttrs.push_back({rewriter.getStringAttr("padH_high"),
                                    rewriter.getI64IntegerAttr(padH_high)});
            paddingAttrs.push_back({rewriter.getStringAttr("padW_low"),
                                    rewriter.getI64IntegerAttr(padW_low)});
            paddingAttrs.push_back({rewriter.getStringAttr("padW_high"),
                                    rewriter.getI64IntegerAttr(padW_high)});

            convOp2->setAttr("exsleratev2.padding",
                             rewriter.getDictionaryAttr(paddingAttrs));

            return;
          }
        }
      }
    }
  });
}

} // namespace mlir::iree_compiler::GlobalOptimization
