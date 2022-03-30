// RUN: iree-opt -iree-llvmcpu-check-ir-before-llvm-conversion %s -verify-diagnostics -split-input-file

func.func @no_static_allocas(%arg0: index) {
  // expected-error @+1 {{expected no static allocations}}
  %0 = memref.alloca(%arg0) : memref<?xf32>
  return
}
