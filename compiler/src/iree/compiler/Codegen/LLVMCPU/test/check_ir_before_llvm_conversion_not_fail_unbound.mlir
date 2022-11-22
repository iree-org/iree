// RUN: iree-opt --iree-llvmcpu-check-ir-before-llvm-conversion --iree-llvmcpu-fail-unbound-dynamic-stack-allocation=false %s --verify-diagnostics -split-input-file

module {
  func.func @dynamic_allocas(%arg0: index) {
    %0 = memref.alloca(%arg0) : memref<?xf32>
    return
  }
}

// -----

#map = affine_map<(d0) -> (-d0, 16384)>
// expected-error @+1 {{expected total size of stack allocation is not greater than 32768 bytes, but got 65536 bytes}}
module {
  func.func @dynamic_big_allocas(%arg0: index, %arg1: index) {
    %0 = affine.min #map(%arg0)
    %1 = memref.alloca(%0, %arg1) : memref<?x?xf32>
    return
  }
}

// -----

// expected-error @+1 {{expected total size of stack allocation is not greater than 32768 bytes, but got 65536 bytes}}
module {
  func.func @mix_static_and_unbound_dynamic_allocas(%arg0: index) {
    %0 = memref.alloca(%arg0) : memref<?x16384xf32>
    return
  }
}
