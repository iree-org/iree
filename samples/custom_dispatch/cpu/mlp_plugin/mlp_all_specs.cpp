

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#include "mlp_all_specs.h.inc"


void test(RewritePatternSet& patterns){
    populateGeneratedPDLLPatterns(patterns);
    patterns.getPDLPatterns().getModule().dump();
}

}  // namespace mlir


// Get PDL with this perhaps?
// int main() {
//     mlir::MLIRContext context;
//     mlir::RewritePatternSet patterns(&context);
//     mlir::test(patterns);
//     return 0;
// }