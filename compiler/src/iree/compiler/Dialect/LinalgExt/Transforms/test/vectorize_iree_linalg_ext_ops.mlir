// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-linalg-ext-vectorize-ops))' --split-input-file %s | FileCheck %s

func.func @map_scatter(
    %input: tensor<4x16x64xf32>, %output: tensor<4x16x64xf32>
) -> tensor<4x16x64xf32> {
  %0 = iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index, %idx1: index, %idx2: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %idx2, %mask : index, index, index, i1
  } : tensor<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
  return %0 : tensor<4x16x64xf32>
}
// CHECK-LABEL: @map_scatter
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]
//  CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9_]+]]
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[INPUT]]
//       CHECK:   %[[MAP_SCATTER:.+]] = iree_linalg_ext.map_scatter
//  CHECK-SAME:     %[[READ]] into %[[OUTPUT]]
//       CHECK:     : vector<4x16x64xf32> into tensor<4x16x64xf32> -> tensor<4x16x64xf32>
//       CHECK:   return %[[MAP_SCATTER]] : tensor<4x16x64xf32>

// -----

func.func @no_vectorize_map_scatter_dynamic(
    %input: tensor<?xf32>, %output: tensor<64xf32>
) -> tensor<64xf32> {
  %0 = iree_linalg_ext.map_scatter %input into %output {
    ^bb0(%idx0: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %mask : index, i1
  } : tensor<?xf32> into tensor<64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
// CHECK-LABEL: @no_vectorize_map_scatter_dynamic
//   CHECK-NOT:   vector
