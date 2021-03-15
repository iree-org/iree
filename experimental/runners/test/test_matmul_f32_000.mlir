// RUN: export M=32 && export N=64 && export K=128 && export ITERS=10 && \
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0

// Analysis kicks in, we can write in %[[C]] and no spurious alloc/copies are inserted.
//  CHECK-NEXT:   linalg.fill(%[[C]], %{{.*}}) : memref<32x64xf32>, f32
//  CHECK-NEXT:   linalg.matmul ins(%[[A]], %[[B]] : memref<32x128xf32>, memref<128x64xf32>) outs(%[[C]] : memref<32x64xf32>)
//  CHECK-NEXT:   return

// CHECK-LABEL: func @main(
//   CHECK-DAG:   %[[f0:.*]] = constant 0.0
//   CHECK-DAG:   %[[f1:.*]] = constant 1.0
//   CHECK-DAG:   %[[f2:.*]] = constant 2.0
//   CHECK-DAG:   alloc() : memref<32x128xf32>
//   CHECK-DAG:   alloc() : memref<128x64xf32>
//   CHECK-DAG:   alloc() : memref<32x64xf32>
//   CHECK-DAG:   linalg.fill(%[[A:.*]], %[[f1]]) : memref<32x128xf32>, f32
//   CHECK-DAG:   linalg.fill(%[[B:.*]], %[[f2]]) : memref<128x64xf32>, f32
//   CHECK-DAG:   linalg.fill(%[[C:.*]], %[[f0]]) : memref<32x64xf32>, f32

// On the caller side, we do not (yet) determine that the scf.for operand used in
// iterative calls to init_and_matmul can all be made in place.
// So an extra alloc + copy is performed form which the final result is read.
//       CHECK:   %[[RES:.*]] = alloc() : memref<32x64xf32>
//       CHECK:   call @rtclock() : () -> f64
//       CHECK:   scf.for %{{.*}} {
//  CHECK-NEXT:     call @init_and_matmul(%[[A]], %[[B]], %[[C]]) : (memref<32x128xf32>, memref<128x64xf32>, memref<32x64xf32>) -> ()
//  CHECK-NEXT:   }
//       CHECK:   linalg.copy(%[[C]], %[[RES]]) : memref<32x64xf32>, memref<32x64xf32>
//       CHECK:   call @rtclock() : () -> f64
//       CHECK:   vector.transfer_read %[[RES]]
