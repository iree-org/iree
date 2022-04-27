// RUN: iree-opt -iree-llvmcpu-check-ir-before-llvm-conversion %s -verify-diagnostics -split-input-file

module {
func.func @no_dynamic_allocas(%arg0: index) {
  // expected-error @+1 {{expected no stack allocations with dynamic shapes}}
  %0 = memref.alloca(%arg0) : memref<?xf32>
  return
}
}

// -----

// expected-error @+1 {{expected total size of stack allocation is smaller than 16 KB}}
module {
func.func @big_allocas(%arg0: index) {
  %0 = memref.alloca() : memref<65536xi32>
  return
}
}
