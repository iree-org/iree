// RUN: iree-opt --iree-llvmcpu-check-ir-before-llvm-conversion --iree-llvmcpu-fail-unbound-dynamic-stack-allocation %s --verify-diagnostics -split-input-file

module {
  func.func @dynamic_allocas(%arg0: index) {
    // expected-error @+1 {{expected no stack allocations without upper bound shapes}}
    %0 = memref.alloca(%arg0) : memref<?xf32>
    return
  }
}
