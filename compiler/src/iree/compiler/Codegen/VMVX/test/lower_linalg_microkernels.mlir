// RUN: iree-opt --pass-pipeline="func.func(iree-vmvx-lower-linalg-microkernels, canonicalize, cse)" %s | FileCheck %s

// Non-identity layouts cannot be resolved at function boundaries, and having
// one exercises a corner case to ensure that we are not making IR modifications
// prior to matching which cause non-convergence.
// CHECK-LABEL: @subview_indexing_2d_indexing_failure
func.func @subview_indexing_2d_indexing_failure(%arg0 : memref<384x128xf32>, %arg1 : memref<128x384xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>, %arg2 : index, %arg3 : index) {
  %6 = memref.subview %arg0[%arg2, %arg3] [64, 64] [1, 1] : memref<384x128xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
  %7 = memref.subview %arg1[%arg3, %arg2] [64, 64] [1, 1] : memref<128x384xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
  // A non-broadcasting 2d copy.
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%7 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>)
    outs(%6 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) {
  ^bb0(%arg4: f32, %arg5: f32):
    linalg.yield %arg4 : f32
  }
  func.return
}

// Verifies the indexing math generated in order to resolve subviews to 1D.
// This incidentally also verifies vmvx.copy (non-transposed) lowering.
// CHECK-LABEL: @subview_indexing_2d
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
//   CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
//   CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//   CHECK-DAG: %[[I0:.*]] = arith.muli %arg2, %[[C128]] : index
//   CHECK-DAG: %[[I1:.*]] = arith.addi %[[I0]], %arg3 : index
//   CHECK-DAG: %[[I2:.*]] = arith.muli %arg3, %[[C384]] : index
//   CHECK-DAG: %[[I3:.*]] = arith.addi %[[I2]], %arg2 : index
//   CHECK-DAG: %[[ARG0_1D:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<384x128xf32> into memref<49152xf32>
//   CHECK-DAG: %[[ARG1_1D:.*]] = memref.collapse_shape %arg1 {{\[}}[0, 1]] : memref<128x384xf32> into memref<49152xf32>
//       CHECK: vmvx.copy in(%[[ARG1_1D]] offset %[[I3]] strides[%[[C384]], %[[C1]]] : memref<49152xf32>)
//  CHECK-SAME:   out(%[[ARG0_1D]] offset %[[I1]] strides[%[[C128]], %[[C1]]] : memref<49152xf32>)
//  CHECK-SAME:   sizes(%[[C64]], %[[C64]])
func.func @subview_indexing_2d(%arg0 : memref<384x128xf32>, %arg1 : memref<128x384xf32>, %arg2 : index, %arg3 : index) {
  %6 = memref.subview %arg0[%arg2, %arg3] [64, 64] [1, 1] : memref<384x128xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
  %7 = memref.subview %arg1[%arg3, %arg2] [64, 64] [1, 1] : memref<128x384xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
  // A non-broadcasting 2d copy.
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%7 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>)
    outs(%6 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) {
  ^bb0(%arg4: f32, %arg5: f32):
    linalg.yield %arg4 : f32
  }
  func.return
}

// Verifies that 2d generic with swapped dims lowers to vmvx.copy with swapped
// strides.
// CHECK-LABEL: @generic_2d_transposed_to_copy
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
//   CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
//       CHECK: vmvx.copy in({{.*}} offset {{.*}} strides[%[[C1]], %[[C384]]] : memref<49152xf32>)
//  CHECK-SAME:   out({{.*}} offset {{.*}} strides[%[[C128]], %[[C1]]] : memref<49152xf32>) sizes({{.*}})
func.func @generic_2d_transposed_to_copy(%arg0 : memref<384x128xf32>, %arg1 : memref<128x384xf32>, %arg2 : index, %arg3 : index) {
  %6 = memref.subview %arg0[%arg2, %arg3] [64, 64] [1, 1] : memref<384x128xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
  %7 = memref.subview %arg1[%arg3, %arg2] [64, 64] [1, 1] : memref<128x384xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
  // A transposed 2d copy.
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%7 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>)
    outs(%6 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) {
  ^bb0(%arg4: f32, %arg5: f32):
    linalg.yield %arg4 : f32
  }
  func.return
}

// CHECK-LABEL: @fill2d
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
//   CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
//       CHECK: %[[ARG0_1D:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<384x128xf32> into memref<49152xf32>
// CHECK: vmvx.fill2d scalar(%arg1 : f32) out(%[[ARG0_1D]] offset %[[C0]] row_stride %[[C128]] : memref<49152xf32>) sizes(%[[C384]], %[[C128]])
func.func @fill2d(%arg0 : memref<384x128xf32>, %arg1 : f32) {
  linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<384x128xf32>)
  func.return
}

// CHECK-LABEL: @matmul_row_major
//   CHECK-DAG: %[[SCALE:.*]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
//   CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//   CHECK-DAG: %[[ARG0_1D:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<64x64xf32> into memref<4096xf32>
//   CHECK-DAG: %[[ARG1_1D:.*]] = memref.collapse_shape %arg1 {{\[}}[0, 1]] : memref<64x384xf32> into memref<24576xf32>
//   CHECK-DAG: %[[ARG2_1D:.*]] = memref.collapse_shape %arg2 {{\[}}[0, 1]] : memref<384x64xf32> into memref<24576xf32>
//       CHECK: vmvx.matmul lhs(%[[ARG1_1D]] offset %c0 row_stride %c384 : memref<24576xf32>)
//  CHECK-SAME:   rhs(%[[ARG2_1D]] offset %[[C0]] row_stride %[[C64]] : memref<24576xf32>)
//  CHECK-SAME:   out(%[[ARG0_1D]] offset %[[C0]] row_stride %[[C64]] : memref<4096xf32>)
//  CHECK-SAME:   mnk(%[[C64]], %[[C64]], %[[C384]]) scale(%[[SCALE]] : f32, %[[SCALE]] : f32)
//  CHECK-SAME:   flags(0)
func.func @matmul_row_major(%arg0 : memref<64x64xf32>, %arg1 : memref<64x384xf32>, %arg2 : memref<384x64xf32>) {
  linalg.matmul
      ins(%arg1, %arg2 : memref<64x384xf32>, memref<384x64xf32>)
      outs(%arg0 : memref<64x64xf32>)
  func.return
}

// CHECK-LABEL: @addf2d_rank_broadcast
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
//   CHECK-DAG: %[[ARG0_1D:.*]] = memref.collapse_shape %arg0 {{\[}}[0, 1]] : memref<64x64xf32> into memref<4096xf32>
//       CHECK: vmvx.add lhs(%arg1 offset %[[C0]] strides[%[[C0]], %[[C1]]] : memref<64xf32>)
//  CHECK-SAME:   rhs(%[[ARG0_1D]] offset %[[C0]] strides[%[[C64]], %[[C1]]] : memref<4096xf32>)
//  CHECK-SAME:   out(%[[ARG0_1D]] offset %[[C0]] strides[%[[C64]], %[[C1]]] : memref<4096xf32>)
//  CHECK-SAME:   sizes(%[[C64]], %[[C64]])
func.func @addf2d_rank_broadcast(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.addf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @addf0d
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[ARG1_1D:.*]] = memref.expand_shape %arg1 [] : memref<f32> into memref<1xf32>
//       CHECK: vmvx.add lhs(%[[ARG1_1D]] offset %[[C0]] strides[%[[C0]], %[[C0]]] : memref<1xf32>)
//  CHECK-SAME: rhs(%arg0 offset %c0 strides[%[[C0]], %[[C1]]] : memref<2xf32>)
//  CHECK-SAME: out(%arg0 offset %[[C0]] strides[%[[C0]], %[[C1]]] : memref<2xf32>) sizes(%[[C1]], %[[C2]]) : f32
func.func @addf0d(%arg0 : memref<2xf32>, %arg1 : memref<f32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%arg1 : memref<f32>) outs(%arg0 : memref<2xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.addf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}
