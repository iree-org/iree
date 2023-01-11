#ifndef IREE_COMPILER_TOOLS_INIT_IREE_H_
#define IREE_COMPILER_TOOLS_INIT_IREE_H_

#include "llvm/Support/InitLLVM.h"

namespace mlir::iree_compiler {

// Helper to initialize LLVM while setting IREE specific help and versions.
class InitIree {
 public:
  InitIree(int &argc, char **&argv);

 private:
  llvm::InitLLVM init_llvm_;
};

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_TOOLS_INIT_IREE_H_
