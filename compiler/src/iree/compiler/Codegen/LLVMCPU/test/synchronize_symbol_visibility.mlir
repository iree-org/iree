// RUN: iree-opt --iree-llvmcpu-synchronize-symbol-visibility %s | FileCheck %s

// CHECK-LABEL: llvm.func internal @internal_fn() attributes {sym_visibility = "private"}
llvm.func internal @internal_fn() {
  llvm.return
}
