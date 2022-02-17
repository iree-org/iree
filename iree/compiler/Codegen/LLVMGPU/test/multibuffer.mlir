// RUN: iree-opt %s -allow-unregistered-dialect -iree-llvmgpu-multi-buffering -split-input-file | FileCheck %s

// TODO: those tests will be moved upstream once the rest of multi-buffer lands.
// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (((d0 - d1) floordiv d2) mod 5)>

// CHECK-LABEL: func @multi_buffer
func @multi_buffer(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %[[C1]]
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]], %[[C1]], %[[C3]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, #[[$MAP0]]>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, #[[$MAP0]]>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
    "some_use"(%0) : (memref<4x128xf32>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}

// -----
// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (((d0 - d1) floordiv d2) mod 5)>

// CHECK-LABEL: func @multi_buffer_subview_use
func @multi_buffer_subview_use(%a: memref<1024x1024xf32>) {
// CHECK-DAG: %[[A:.*]] = memref.alloc() : memref<5x4x128xf32>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  %0 = memref.alloc() : memref<4x128xf32>
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
// CHECK: scf.for %[[IV:.*]] = %[[C1]]
  scf.for %arg2 = %c1 to %c1024 step %c3 {
// CHECK: %[[I:.*]] = affine.apply #[[$MAP1]](%[[IV]], %[[C1]], %[[C3]])
// CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0, 0] [1, 4, 128] [1, 1, 1] : memref<5x4x128xf32> to memref<4x128xf32, #[[$MAP0]]>
   %1 = memref.subview %a[%arg2, 0] [4, 128] [1, 1] :
    memref<1024x1024xf32> to memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
// CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4x128xf32, #{{.*}}> to memref<4x128xf32, #[[$MAP0]]>
   memref.copy %1, %0 : memref<4x128xf32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<4x128xf32>
// CHECK: %[[SV1:.*]] = memref.subview %[[SV]][0, 1] [4, 127] [1, 1] : memref<4x128xf32, #[[$MAP0]]> to memref<4x127xf32, #[[$MAP0]]>
   %s = memref.subview %0[0, 1] [4, 127] [1, 1] :
      memref<4x128xf32> to memref<4x127xf32, affine_map<(d0, d1) -> (d0 * 128 + d1 + 1)>>
// CHECK: "some_use"(%[[SV1]]) : (memref<4x127xf32, #[[$MAP0]]>) -> ()
   "some_use"(%s) : (memref<4x127xf32, affine_map<(d0, d1) -> (d0 * 128 + d1 + 1)>>) -> ()
// CHECK: "some_use"(%[[SV]]) : (memref<4x128xf32, #[[$MAP0]]>) -> ()
   "some_use"(%0) : (memref<4x128xf32>) -> ()
  }
  return
}
