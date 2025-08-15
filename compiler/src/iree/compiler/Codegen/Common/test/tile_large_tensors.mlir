// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-tile-large-tensors, canonicalize, cse))" | \
// RUN:   FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_generic(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>, %5: tensor<64x256xf32>) -> tensor<64x256xf32> {
  %6 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
    } ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%5 : tensor<64x256xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<64x256xf32>
  return %6 : tensor<64x256xf32>
}

// CHECK-LABEL: func.func @simple_generic
//       CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c1
//       CHECK:     scf.for %{{.*}} = %c0 to %c256 step %c64
//       CHECK:       linalg.generic {{.*}} outs({{.*}}: tensor<1x64xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_destination(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %7 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @fuse_destination
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xf32>
//       CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c1
//       CHECK:     linalg.fill {{.*}} -> tensor<1x64xf32>

// Additionally verify that reduction dimensions do not get tiled.
//       CHECK:     linalg.matmul ins({{.*}}: tensor<1x64xf32>, tensor<64x64xf32>)

// -----

func.func @in_nested_region(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>, %5: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %c64 step %c8 iter_args(%arg1 = %5) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_0 = tensor.extract_slice %4[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %7 = linalg.add
      ins(%extracted_slice, %extracted_slice_0 : tensor<64x8xf32>, tensor<64x8xf32>)
      outs(%extracted_slice_1 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %insert = tensor.insert_slice %7 into %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
    scf.yield %insert : tensor<64x64xf32>
  }
  return %6 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @in_nested_region
//       CHECK:   scf.for
//       CHECK:     scf.for %{{.*}} = %c0 to %c64 step %c1
//       CHECK:       linalg.add {{.*}} -> tensor<1x8xf32>

// -----

func.func @multiple_use_tilable_op(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>) -> (tensor<64x256xf32>, tensor<64x256xf32>) {
  %empty = tensor.empty() : tensor<64x256xf32>
  %6 = linalg.add
    ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>)
    outs(%empty : tensor<64x256xf32>) -> tensor<64x256xf32>
  %7 = linalg.exp
    ins(%6 : tensor<64x256xf32>)
    outs(%empty : tensor<64x256xf32>) -> tensor<64x256xf32>
  return %6, %7 : tensor<64x256xf32>, tensor<64x256xf32>
}

// CHECK-LABEL: func.func @multiple_use_tilable_op
//       CHECK:   %[[ADD_TILING:.+]] = scf.for
//       CHECK:     linalg.add {{.*}} -> tensor<1x64xf32>
//       CHECK:   %[[EXP_TILING:.+]] = scf.for
//       CHECK:     %[[FUSED_ADD:.+]] = linalg.add {{.*}} -> tensor<1x64xf32>
//       CHECK:     linalg.exp ins(%[[FUSED_ADD]]
//       CHECK:   return %[[ADD_TILING]], %[[EXP_TILING]]

// -----

func.func @no_tile_transpose(%arg0: tensor<64x256xf32>) -> tensor<256x64xf32> {
  %empty = tensor.empty() : tensor<256x64xf32>
  %0 = linalg.transpose
    ins(%arg0 : tensor<64x256xf32>)
    outs(%empty : tensor<256x64xf32>) permutation = [1, 0]
  return %0 : tensor<256x64xf32>
}

// CHECK-LABEL: func.func @no_tile_transpose
//   CHECK-NOT:   scf.for
//       CHECK:   %[[T:.+]] = linalg.transpose
//       CHECK:   return %[[T]]

// -----

func.func @no_tile_copy(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
  %empty = tensor.empty() : tensor<64x256xf32>
  %0 = linalg.copy
    ins(%arg0 : tensor<64x256xf32>)
    outs(%empty : tensor<64x256xf32>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}

// CHECK-LABEL: func.func @no_tile_copy
//   CHECK-NOT:   scf.for
//       CHECK:   %[[COPY:.+]] = linalg.copy
//       CHECK:   return %[[COPY]]

// -----

func.func @no_tile_fill(%arg0: f32) -> tensor<64x256xf32> {
  %empty = tensor.empty() : tensor<64x256xf32>
  %0 = linalg.fill
    ins(%arg0 : f32)
    outs(%empty : tensor<64x256xf32>) -> tensor<64x256xf32>
  return %0 : tensor<64x256xf32>
}

// CHECK-LABEL: func.func @no_tile_fill
//   CHECK-NOT:   scf.for
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   return %[[FILL]]

// -----

func.func @dynamic_reduction_dim(%arg0: tensor<?x?xf32>, %arg1: tensor<1x?xf32>, %arg2: tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<1x?xf32>)
  outs(%arg2 : tensor<?x1xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: func.func @dynamic_reduction_dim
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?xf32>
//       CHECK:   scf.for %[[I:.+]] = %c0 to %[[DIM]] step %c1
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%{{.*}}: tensor<1x?xf32>, tensor<1x?xf32>)
//  CHECK-SAME:       outs(%{{.*}}: tensor<1x1xf32>)

// -----

#map = affine_map<() -> ()>
func.func @skip_empty_parallel_loops(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = []
    } ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%arg2 : tensor<f32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @skip_empty_parallel_loops
//   CHECK-NOT:   scf.for
