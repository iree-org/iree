// RUN: iree-opt --iree-llvmcpu-check-ir-before-llvm-conversion %s --verify-diagnostics -split-input-file

module {
  func.func @dynamic_allocas(%arg0: index) {
    %0 = memref.alloca(%arg0) : memref<?xf32>
    return
  }
}

// -----

// expected-error @+1 {{expected total size of stack allocation is not greater than 32768 bytes, but got 65536 bytes}}
module {
  func.func @mix_static_dynamic_allocas(%arg0: index) {
    %0 = memref.alloca(%arg0) : memref<?x16384xf32>
    return
  }
}
