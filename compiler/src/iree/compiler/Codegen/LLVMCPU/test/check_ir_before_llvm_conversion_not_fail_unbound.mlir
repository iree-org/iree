// RUN: iree-opt --iree-llvmcpu-check-ir-before-llvm-conversion=fail-on-out-of-bounds=false %s --verify-diagnostics -split-input-file

module {
  func.func @dynamic_allocas(%arg0: index) {
    %0 = memref.alloca(%arg0) : memref<?xf32>
    return
  }
}
// CHECK-LABEL: func @dynamic_allocas(

// -----

#map = affine_map<(d0) -> (-d0, 16384)>
module {
  func.func @dynamic_big_allocas(%arg0: index, %arg1: index) {
    %0 = affine.min #map(%arg0)
    %1 = memref.alloca(%0, %arg1) : memref<?x?xf32>
    return
  }
}
// CHECK-LABEL: func @dynamic_big_allocas(

// -----

module {
  func.func @mix_static_and_unbound_dynamic_allocas(%arg0: index) {
    %0 = memref.alloca(%arg0) : memref<?x16384xf32>
    return
  }
}
// CHECK-LABEL: func @mix_static_and_unbound_dynamic_allocas(

// -----

module {
  func.func @scalable_alloca() {
    %c16384 = arith.constant 16384 : index
    %vscale = vector.vscale
    %c16384_vscale = arith.muli %vscale, %c16384 : index
    %0 = memref.alloca(%c16384_vscale) : memref<?xf32>
    return
  }
}
// CHECK-LABEL: func @scalable_alloca(
