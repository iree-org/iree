// RUN: iree-opt --iree-llvmcpu-check-ir-before-llvm-conversion %s --verify-diagnostics -split-input-file

module {
  func.func @dynamic_allocas(%arg0: index) {
    // expected-error @+1 {{expected no stack allocations without upper bound shapes}}
    %0 = memref.alloca(%arg0) : memref<?xf32>
    return
  }
}

// -----

// expected-error @+1 {{expected total size of stack allocation is not greater than 32 KB, but got 65536 bytes}}
module {
  func.func @static_big_allocas(%arg0: index) {
    %0 = memref.alloca() : memref<16384xi32>
    return
  }
}

// -----

#map = affine_map<(d0) -> (-d0, 16384)>
// expected-error @+1 {{expected total size of stack allocation is not greater than 32 KB, but got 65536 bytes}}
module {
  func.func @dynamic_big_allocas(%arg0: index) {
    %0 = affine.min #map(%arg0)
    %1 = memref.alloca(%0) : memref<?xf32>
    return
  }
}
