// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-check-ir-before-llvm-conversion))" %s --verify-diagnostics -split-input-file

func.func @dynamic_allocas(%arg0: index) {
  // expected-error @+1 {{expected no unbounded stack allocations}}
  %0 = memref.alloca(%arg0) : memref<?xf32>
  return
}

// -----

// expected-error @+1 {{exceeded stack allocation limit of 32768 bytes for function. Got 65536 bytes}}
func.func @static_big_allocas(%arg0: index) {
  %0 = memref.alloca() : memref<16384xi32>
  return
}

// -----

#map = affine_map<(d0) -> (-d0, 16384)>
// expected-error @+1 {{exceeded stack allocation limit of 32768 bytes for function. Got 65536 bytes}}
func.func @dynamic_big_allocas(%arg0: index) {
  %0 = affine.min #map(%arg0)
  %1 = memref.alloca(%0) : memref<?xf32>
  return
}

// -----

#map = affine_map<(d0) -> (-d0, 16)>
// expected-error @+1 {{exceeded stack allocation limit of 32768 bytes for function. Got 65536 bytes}}
func.func @mix_static_and_dynamic_allocas(%arg0: index) {
  %0 = affine.min #map(%arg0)
  %1 = memref.alloca(%0) : memref<?x1024xf32>
  return
}

// -----

func.func @non_entry_bb_allocas(%arg0: index) {
  cf.br ^bb1
 ^bb1() :
  // expected-error @+1 {{all stack allocations need to be hoisted to the entry block of the function}}
  %0 = memref.alloca() : memref<16xi32>
  return
}

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloca(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    // expected-error @+1 {{all stack allocations need to be hoisted to the entry block of the function}}
    %1 = memref.alloca(%0) : memref<?xi32>
  }
  return
}
