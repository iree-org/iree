#ifndef IREE_COMPILER_ASSERTINSERTER_PASSES_H_
#define IREE_COMPILER_ASSERTINSERTER_PASSES_H_

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DECL
#include "iree/compiler/AssertInserter/Passes.h.inc"

struct AssertInserterPipelineOptions
    : public PassPipelineOptions<AssertInserterPipelineOptions> {
  PassOptions::Option<bool> warnOnUnknown{
      *this, "warn-on-unknown",
      llvm::cl::desc("Warn on unknown side-effecting operations"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> includeVectorLoadStore{
      *this, "include-vector-load-store",
      llvm::cl::desc(
          "Include vector.load/store operations despite them allowing "
          "out-of-bounds"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> checkEachDim{
      *this, "check-each-dim",
      llvm::cl::desc("Check each dimension individually"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> createSpeculativeFuncs{
      *this, "create-speculative-funcs",
      llvm::cl::desc("Create a function that performs assertions speculatively "
                     "instead of in-place checks"),
      llvm::cl::init(false)};
};

void buildAssertInserterPipeline(mlir::OpPassManager &pm,
                                 const AssertInserterPipelineOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_ASSERTINSERTER_PASSES_H_
