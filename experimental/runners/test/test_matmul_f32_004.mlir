// RUN: export M=32 && export N=64 && export K=128 && export ITERS=10 && \
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=2,4,16 pad hoist-padding=3" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul vectorize vector-contract-lowering=false vectorize-padding" |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0
//   CHECK-NOT:   memref.alloc
//       CHECK:   linalg.fill(%[[C]], %{{.*}}) : memref<32x64xf32>, f32
//   CHECK-DAG:   %[[PACKED_B:.*]] = memref.alloc() : memref<16x8x16x4xf32>
//   CHECK-NOT:   copy
//       CHECK:   scf.for %[[J1:.*]] =
//       CHECK:     %[[PACKED_IDX_B_J1:.*]] = affine.apply
//       CHECK:     scf.for %[[K1:.*]] =
//       CHECK:       %[[PACKED_IDX_B_K1:.*]] = affine.apply
//       CHECK:       memref.subview %[[B]][%[[K1]], %[[J1]]] [16, 4] [1, 1] : memref<128x64xf32> to memref<16x4xf32
// Loop order is I, J, K -> packed_B is J x K x tK x tJ
//       CHECK:       memref.subview %[[PACKED_B]][%[[PACKED_IDX_B_J1]], %[[PACKED_IDX_B_K1]], 0, 0] [1, 1, 16, 4] [1, 1, 1, 1] : memref<16x8x16x4xf32> to memref<16x4xf32
//       CHECK:       linalg.copy
//
//   CHECK-DAG:   %[[PACKED_A:.*]] = memref.alloc() : memref<16x8x2x16xf32>
//       CHECK:   scf.for %[[I2:.*]] =
//       CHECK:     %[[PACKED_IDX_A_I:.*]] = affine.apply
//       CHECK:     scf.for %[[K2:.*]] =
//       CHECK:       %[[PACKED_IDX_A_K:.*]] = affine.apply
//       CHECK:       memref.subview %[[A]][%[[I2]], %[[K2]]] [2, 16] [1, 1] : memref<32x128xf32> to memref<2x16xf32
// Loop order is I, J, K -> packed_A is I x K x tI x tK
//       CHECK:       memref.subview %[[PACKED_A]][%[[PACKED_IDX_A_I]], %[[PACKED_IDX_A_K]], 0, 0] [1, 1, 2, 16] [1, 1, 1, 1] : memref<16x8x2x16xf32> to memref<2x16xf32
//       CHECK:       linalg.copy
//
//       CHECK:   scf.for %[[I:.*]] =
//       CHECK:     %[[PACKED_IDX_I:.*]] = affine.apply
//       CHECK:     scf.for %[[J:.*]] =
//       CHECK:       %[[PACKED_IDX_J:.*]] = affine.apply
//       CHECK:       %[[SVC:.*]] = memref.subview %[[C]]{{.*}} : memref<32x64xf32> to memref<2x4xf32
//       CHECK:       %[[VC:.*]] = vector.transfer_read %[[SVC]]{{.*}}{masked = [false, false]} : memref<2x4xf32{{.*}}>, vector<2x4xf32>
//       CHECK:       scf.for %[[K:.*]] = {{.*}} iter_args(%{{.*}} = %[[VC]]) -> (vector<2x4xf32>)
//       CHECK:         %[[PACKED_IDX_K:.*]] = affine.apply
// Loop order is I, J, K -> packed_A is I x K x tI x tK
//       CHECK:         %[[SVA:.*]] = memref.subview %[[PACKED_A]][%[[PACKED_IDX_I]], %[[PACKED_IDX_K]], 0, 0] [1, 1, 2, 16] [1, 1, 1, 1] : memref<16x8x2x16xf32> to memref<2x16xf32
// Loop order is I, J, K -> packed_B is J x K x tK x tJ
//       CHECK:         %[[SVB:.*]] = memref.subview %[[PACKED_B]][%[[PACKED_IDX_J]], %[[PACKED_IDX_K]], 0, 0] [1, 1, 16, 4] [1, 1, 1, 1] : memref<16x8x16x4xf32> to memref<16x4xf32
//       CHECK:         vector.transfer_read %[[SVA]]{{.*}} {masked = [false, false]} : memref<2x16xf32{{.*}}>, vector<2x16xf32>
//       CHECK:         vector.transfer_read %[[SVB]]{{.*}}, %cst {masked = [false, false]} : memref<16x4xf32{{.*}}>, vector<16x4xf32>
//       CHECK:         %[[RES:.*]] = vector.contract
//       CHECK:         scf.yield %[[RES]] : vector<2x4xf32>
//   CHECK-NOT:         copy
//       CHECK:       }
//       CHECK:       vector.transfer_write %{{.*}}, %[[SVC]]{{.*}}{masked = [false, false]} : vector<2x4xf32>, memref<2x4xf32
//   CHECK-NOT:       copy
//       CHECK:     }
//       CHECK:   }
//   CHECK-NOT:   copy
//   CHECK-DAG:   memref.dealloc %[[PACKED_A]]
//   CHECK-DAG:   memref.dealloc %[[PACKED_B]]
