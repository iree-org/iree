// RUN: export M=128 && export N=128 && export K=128 && export ITERS=10 && \
// RUN: cat %p/matmul_f32_base.mlir | sed 's@${M}@'"$M"'@g'| sed 's@${K}@'"$K"'@g' | sed 's@${N}@'"$N"'@g'| sed 's@${ITERS}@'"$ITERS"'@g' |\

// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

// CHECK-LABEL: func @init_and_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
//       CHECK:   constant 0.0

// Analysis kicks in, we can write in %[[C]] and no spurious alloc/copies are inserted.
//  CHECK-NEXT:   linalg.fill(%[[C]], %{{.*}}) : memref<128x128xf32>, f32
//  CHECK-NEXT:   linalg.matmul ins(%[[A]], %[[B]] : memref<128x128xf32>, memref<128x128xf32>) outs(%[[C]] : memref<128x128xf32>)
//  CHECK-NEXT:   return

// CHECK-LABEL: func @main(
//   CHECK-DAG:   %[[f0:.*]] = constant 0.0
//   CHECK-DAG:   %[[f1:.*]] = constant 1.0
//   CHECK-DAG:   %[[f2:.*]] = constant 2.0
//   CHECK-DAG:   alloc() : memref<128x128xf32>
//   CHECK-DAG:   alloc() : memref<128x128xf32>
//   CHECK-DAG:   alloc() : memref<128x128xf32>
//   CHECK-DAG:   linalg.fill(%[[A:.*]], %[[f1]]) : memref<128x128xf32>, f32
//   CHECK-DAG:   linalg.fill(%[[B:.*]], %[[f2]]) : memref<128x128xf32>, f32
//   CHECK-DAG:   linalg.fill(%[[C:.*]], %[[f0]]) : memref<128x128xf32>, f32
//       CHECK:   call @rtclock() : () -> f64
//  CHECK-NEXT:   scf.for %{{.*}} {
//  CHECK-NEXT:     call @init_and_matmul(%[[A]], %[[B]], %[[C]]) : (memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>) -> ()
//  CHECK-NEXT:   }
//  CHECK-NEXT:   call @rtclock() : () -> f64
