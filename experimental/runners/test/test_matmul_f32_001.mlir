// RUN: export M=32 && export N=64 && export K=128 && export ITERS=10 && \
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_matmul anchor-op=linalg.matmul tile-sizes=4,8,16 vectorize vector-contract-lowering=false" |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0
//   CHECK-NOT:   alloc
//       CHECK:   linalg.fill(%[[C]], %{{.*}}) : memref<32x64xf32>, f32
//   CHECK-NOT:   copy
//       CHECK:   scf.for %[[I:.*]] =
//       CHECK:     scf.for %[[J:.*]] =
//       CHECK:       %[[SVC:.*]] = subview %[[C]]{{.*}} : memref<32x64xf32> to memref<4x8xf32
//       CHECK:       %[[VC:.*]] = vector.transfer_read %[[SVC]]{{.*}}{masked = [false, false]} : memref<4x8xf32{{.*}}>, vector<4x8xf32>
//       CHECK:       scf.for %[[K:.*]] = {{.*}} iter_args(%{{.*}} = %[[VC]]) -> (vector<4x8xf32>)
//       CHECK:         %[[SVA:.*]] = subview %[[A]][%[[I]], %[[K]]] [4, 16] [1, 1] : memref<32x128xf32> to memref<4x16xf32
//       CHECK:         %[[SVB:.*]] = subview %[[B]][%[[K]], %[[J]]] [16, 8] [1, 1] : memref<128x64xf32> to memref<16x8xf32
//       CHECK:         vector.transfer_read %[[SVA]]{{.*}} {masked = [false, false]} : memref<4x16xf32{{.*}}>, vector<4x16xf32>
//       CHECK:         vector.transfer_read %[[SVB]]{{.*}}, %cst {masked = [false, false]} : memref<16x8xf32{{.*}}>, vector<16x8xf32>
//       CHECK:         vector.contract
//       CHECK:         scf.yield %{{.*}} : vector<4x8xf32>
//       CHECK:       }
//       CHECK:       vector.transfer_write %{{.*}}, %[[SVC]]{{.*}}{masked = [false, false]} : vector<4x8xf32>, memref<4x8xf32
//   CHECK-NOT:       copy
//       CHECK:     }
//       CHECK:   }
//   CHECK-NOT:   copy

// CHECK-LABEL: func @main(
