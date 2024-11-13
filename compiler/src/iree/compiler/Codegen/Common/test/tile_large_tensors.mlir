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

func.func @multiple_use_tilable_op(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>) -> (tensor<64x256xf32>, tensor<256x64xf32>) {
  %add_empty = tensor.empty() : tensor<64x256xf32>
  %6 = linalg.add
    ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>)
    outs(%add_empty : tensor<64x256xf32>) -> tensor<64x256xf32>
  %transpose_empty = tensor.empty() : tensor<256x64xf32>
  %7 = linalg.transpose
    ins(%6 : tensor<64x256xf32>)
    outs(%transpose_empty : tensor<256x64xf32>) permutation = [1, 0]
  return %6, %7 : tensor<64x256xf32>, tensor<256x64xf32>
}

// CHECK-LABEL: func.func @multiple_use_tilable_op
//       CHECK:   %[[ADD_TILING:.+]] = scf.for
//       CHECK:     linalg.add {{.*}} -> tensor<1x64xf32>
//       CHECK:   %[[T_TILING:.+]] = scf.for
//       CHECK:     %[[FUSED_ADD:.+]] = linalg.add {{.*}} -> tensor<64x1xf32>
//       CHECK:     linalg.transpose ins(%[[FUSED_ADD]]
//       CHECK:   return %[[ADD_TILING]], %[[T_TILING]]
