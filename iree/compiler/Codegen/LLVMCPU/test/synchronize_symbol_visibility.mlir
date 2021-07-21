// RUN: iree-opt -iree-llvmcpu-synchronize-symbol-visibility %s | IreeFileCheck %s

// CHECK: llvm.func internal @internal_fn() attributes {sym_visibility = "private"}
llvm.func internal @internal_fn() {
  llvm.return
}
