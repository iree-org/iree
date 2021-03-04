// RUN: export M=128 && export N=128 && export K=128 && export ITERS=10 &&\
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=4,8,16 pad hoist-padding=2" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul vectorize vector-contract-lowering=false vectorize-padding" |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0
//   CHECK-NOT:   alloc
//       CHECK:   linalg.fill(%[[C]], %{{.*}}) : memref<128x128xf32>, f32
//   CHECK-DAG:   %[[PACKED_A:.*]] = alloc() : memref<8x4x16xf32>
//   CHECK-DAG:   %[[PACKED_B:.*]] = alloc() : memref<8x16x16x8xf32>
//   CHECK-NOT:   copy
//       CHECK:   scf.for %[[I:.*]] =
//       CHECK:     scf.for %[[J1:.*]] =
//       CHECK:       %[[PACKED_IDX_B_J1:.*]] = affine.apply
//       CHECK:       scf.for %[[K1:.*]] =
//       CHECK:         %[[PACKED_IDX_B_K1:.*]] = affine.apply
//       CHECK:         subview %[[B]][%[[K1]], %[[J1]]] [16, 8] [1, 1] : memref<128x128xf32> to memref<16x8xf32
//       CHECK:         subview %[[PACKED_B]][%[[PACKED_IDX_B_J1]], %[[PACKED_IDX_B_K1]], 0, 0] [1, 1, 16, 8] [1, 1, 1, 1] : memref<8x16x16x8xf32> to memref<16x8xf32
//       CHECK:         linalg.copy
//       CHECK:     scf.for %[[K2:.*]] =
//       CHECK:       %[[PACKED_IDX_A:.*]] = affine.apply
//       CHECK:       subview %[[A]][%[[I]], %[[K2]]] [4, 16] [1, 1] : memref<128x128xf32> to memref<4x16xf32
//       CHECK:       subview %[[PACKED_A]][%[[PACKED_IDX_A]], 0, 0] [1, 4, 16] [1, 1, 1] : memref<8x4x16xf32> to memref<4x16xf32
//       CHECK:       linalg.copy
//       CHECK:     scf.for %[[J:.*]] =
//       CHECK:       %[[PACKED_IDX_J:.*]] = affine.apply
//       CHECK:       %[[SVC:.*]] = subview %[[C]]{{.*}} : memref<128x128xf32> to memref<4x8xf32
//       CHECK:       %[[VC:.*]] = vector.transfer_read %[[SVC]]{{.*}}{masked = [false, false]} : memref<4x8xf32{{.*}}>, vector<4x8xf32>
//       CHECK:       scf.for %[[K:.*]] = {{.*}} iter_args(%{{.*}} = %[[VC]]) -> (vector<4x8xf32>)
//       CHECK:         %[[PACKED_IDX_K:.*]] = affine.apply
//       CHECK:         %[[SVA:.*]] = subview %[[PACKED_A]][%[[PACKED_IDX_K]], 0, 0] [1, 4, 16] [1, 1, 1] : memref<8x4x16xf32> to memref<4x16xf32
//       CHECK:         %[[SVB:.*]] = subview %[[PACKED_B]][%[[PACKED_IDX_K]], %[[PACKED_IDX_J]], 0, 0] [1, 1, 16, 8] [1, 1, 1, 1] : memref<8x16x16x8xf32> to memref<16x8xf32
//       CHECK:         vector.transfer_read %[[SVA]]{{.*}} {masked = [false, false]} : memref<4x16xf32{{.*}}>, vector<4x16xf32>
//       CHECK:         vector.transfer_read %[[SVB]]{{.*}}, %cst {masked = [false, false]} : memref<16x8xf32{{.*}}>, vector<16x8xf32>
//       CHECK:         %[[RES:.*]] = vector.contract
//       CHECK:         scf.yield %[[RES]] : vector<4x8xf32>
//   CHECK-NOT:         copy
//       CHECK:       }
//       CHECK:       vector.transfer_write %{{.*}}, %[[SVC]]{{.*}}{masked = [false, false]} : vector<4x8xf32>, memref<4x8xf32
//   CHECK-NOT:       copy
//       CHECK:     }
//       CHECK:   }
//   CHECK-NOT:   copy
//   CHECK-DAG:   dealloc %[[PACKED_A]]
//   CHECK-DAG:   dealloc %[[PACKED_B]]
