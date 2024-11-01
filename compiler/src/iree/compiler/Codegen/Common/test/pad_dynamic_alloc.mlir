// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-pad-dynamic-alloc))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: dynamic_alloc
func.func @dynamic_alloc(%id : index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %dim = affine.min affine_map<()[s0] -> (s0 * -64 + 7, 64)>()[%id]
// CHECK: %[[A:.*]] = memref.alloc() : memref<1x64x32xf32, 3>
// CHECK: %[[S:.*]] = memref.subview %[[A]][0, 0, 0] [1, %{{.*}}, 32] [1, 1, 1] : memref<1x64x32xf32, 3> to memref<1x?x32xf32, strided<[2048, 32, 1]>, 3>
  %0 = memref.alloc(%dim) : memref<1x?x32xf32, 3>
// CHECK: vector.store %{{.*}}, %[[S]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<1x?x32xf32, strided<[2048, 32, 1]>, 3>, vector<4xf32>
  vector.store %cst, %0[%c0, %c0, %c0] : memref<1x?x32xf32, 3>, vector<4xf32>
  return
}

// -----

// CHECK-LABEL: dynamic_alloc_max_0
func.func @dynamic_alloc_max_0(%id : index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %dim = affine.min affine_map<()[s0] -> (s0 * -64 + 7, 64)>()[%id]
  %dim1 = affine.max affine_map<()[s0] -> (s0, 0)>()[%dim]
// CHECK: %[[A:.*]] = memref.alloc() : memref<1x64x32xf32, 3>
// CHECK: %[[S:.*]] = memref.subview %[[A]][0, 0, 0] [1, %{{.*}}, 32] [1, 1, 1] : memref<1x64x32xf32, 3> to memref<1x?x32xf32, strided<[2048, 32, 1]>, 3>
  %0 = memref.alloc(%dim1) : memref<1x?x32xf32, 3>
// CHECK: vector.store %{{.*}}, %[[S]][%{{.*}}, %{{.*}}, %{{.*}}] : memref<1x?x32xf32, strided<[2048, 32, 1]>, 3>, vector<4xf32>
  vector.store %cst, %0[%c0, %c0, %c0] : memref<1x?x32xf32, 3>, vector<4xf32>
  return
}

// -----

func.func @dynamic_bound_alloc(%id : index) {
  %0 = util.assume.int %id<umin = 0, umax =  4088> : index
  %1 = memref.alloc(%0) : memref<?xf32, 3>
  return
}
// CHECK-LABEL: func @dynamic_bound_alloc(
//       CHECK:   %alloc = memref.alloc() : memref<4088xf32, 3>
