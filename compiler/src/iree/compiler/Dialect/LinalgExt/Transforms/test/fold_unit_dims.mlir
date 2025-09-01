// RUN: iree-opt --pass-pipeline="builtin.module(iree-linalg-ext-fold-unit-extent-dims{use-reshapes=false})" %s --split-input-file --mlir-print-local-scope | FileCheck %s --check-prefix=SLICE
// RUN: iree-opt --pass-pipeline="builtin.module(iree-linalg-ext-fold-unit-extent-dims{use-reshapes=true}, canonicalize)" %s --split-input-file --mlir-print-local-scope | FileCheck %s --check-prefix=RESHAPE

util.func public @gather_unit_batch_dims(%source: tensor<4x4x4x4xf16>, %indices: tensor<4x1x2xi64>) -> tensor<4x1x4x4xf16> {
  %empty = tensor.empty() : tensor<4x1x4x4xf16>
  %0 = iree_linalg_ext.gather dimension_map = [0, 1]
          ins(%source, %indices: tensor<4x4x4x4xf16>, tensor<4x1x2xi64>)
          outs(%empty: tensor<4x1x4x4xf16>) -> tensor<4x1x4x4xf16>
  util.return %0 : tensor<4x1x4x4xf16>
}
// SLICE-LABEL: util.func public @gather_unit_batch_dims
//       SLICE:   %[[INDICES:.+]] = tensor.extract_slice %{{.*}}[0, 0, 0] [4, 1, 2] [1, 1, 1] : tensor<4x1x2xi64> to tensor<4x2xi64>
//       SLICE:   %[[OUTPUT:.+]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0] [4, 1, 4, 4] [1, 1, 1, 1] : tensor<4x1x4x4xf16> to tensor<4x4x4xf16>
//       SLICE:   %[[RESULT:.+]] = iree_linalg_ext.gather
//  SLICE-SAME:     -> tensor<4x4x4xf16>
//       SLICE:   %[[EXPANDED_RESULT:.+]] = tensor.insert_slice %[[RESULT]] into %{{.*}}[0, 0, 0, 0] [4, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4x4xf16> into tensor<4x1x4x4xf16>

// RESHAPE-LABEL: util.func public @gather_unit_batch_dims
//       RESHAPE:   %[[INDICES:.+]] = tensor.collapse_shape %{{.*}} : tensor<4x1x2xi64> into tensor<4x2xi64>
//       RESHAPE:   %[[OUTPUT:.+]] = tensor.collapse_shape %{{.*}} : tensor<4x1x4x4xf16> into tensor<4x4x4xf16>
//       RESHAPE:   %[[RESULT:.+]] = iree_linalg_ext.gather
//  RESHAPE-SAME:     -> tensor<4x4x4xf16>
//       RESHAPE:   tensor.expand_shape %[[RESULT]] {{.*}} : tensor<4x4x4xf16> into tensor<4x1x4x4xf16>

// -----

util.func public @gather_batch_and_slice_dims(%source: tensor<4x4x1x4xf16>, %indices: tensor<4x1x2xi64>) -> tensor<4x1x1x4xf16> {
  %empty = tensor.empty() : tensor<4x1x1x4xf16>
  %0 = iree_linalg_ext.gather dimension_map = [0, 1]
          ins(%source, %indices: tensor<4x4x1x4xf16>, tensor<4x1x2xi64>)
          outs(%empty: tensor<4x1x1x4xf16>) -> tensor<4x1x1x4xf16>
  util.return %0 : tensor<4x1x1x4xf16>
}
// SLICE-LABEL: util.func public @gather_batch_and_slice_dims
//  SLICE-SAME:   %[[SOURCE:[a-zA-Z0-9]+]]: tensor<4x4x1x4xf16>
//  SLICE-SAME:   %[[INDICES:[a-zA-Z0-9]+]]: tensor<4x1x2xi64>
//       SLICE:   %[[OUTPUT:.+]] = tensor.empty() : tensor<4x1x1x4xf16>
//       SLICE:   %[[INDICES_BATCH:.+]] = tensor.extract_slice %[[INDICES]]
//  SLICE-SAME:     tensor<4x1x2xi64> to tensor<4x2xi64>
//       SLICE:   %[[OUTPUT_BATCH:.+]] = tensor.extract_slice %[[OUTPUT]]
//  SLICE-SAME:     tensor<4x1x1x4xf16> to tensor<4x1x4xf16>
//       SLICE:   %[[SOURCE_SLICE:.+]] = tensor.extract_slice %[[SOURCE]]
//  SLICE-SAME:   tensor<4x4x1x4xf16> to tensor<4x4x4xf16>
//       SLICE:   %[[OUTPUT_SLICE:.+]] = tensor.extract_slice %[[OUTPUT_BATCH]]
//  SLICE-SAME:     tensor<4x1x4xf16> to tensor<4x4xf16>
//       SLICE:   iree_linalg_ext.gather
//  SLICE-SAME:     ins(%[[SOURCE_SLICE]], %[[INDICES_BATCH]]
//  SLICE-SAME:     outs(%[[OUTPUT_SLICE]]

// RESHAPE-LABEL: util.func public @gather_batch_and_slice_dims
//       RESHAPE:   %[[INDICES:.+]] = tensor.collapse_shape %{{.*}} : tensor<4x1x2xi64> into tensor<4x2xi64>
//       RESHAPE:   %[[SOURCE:.+]] = tensor.collapse_shape %{{.*}} : tensor<4x4x1x4xf16> into tensor<4x4x4xf16>
//       RESHAPE:   %[[OUTPUT:.+]] = tensor.collapse_shape %{{.*}} : tensor<4x1x1x4xf16> into tensor<4x4xf16>
//       RESHAPE:   %[[RESULT:.+]] = iree_linalg_ext.gather
//  RESHAPE-SAME:     -> tensor<4x4xf16>
//       RESHAPE:   tensor.expand_shape %[[RESULT]] {{.*}} : tensor<4x4xf16> into tensor<4x1x1x4xf16>

// -----

util.func public @scatter_batch_and_slice_dims(%slice: tensor<4x1x1x4xf16>, %indices: tensor<4x1x2xi64>) -> tensor<4x4x1x4xf16> {
  %empty = tensor.empty() : tensor<4x4x1x4xf16>
  %0 = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
          ins(%slice, %indices: tensor<4x1x1x4xf16>, tensor<4x1x2xi64>)
          outs(%empty: tensor<4x4x1x4xf16>){
  ^bb0(%in : f16, %out : f16):
    iree_linalg_ext.yield %in : f16
  }-> tensor<4x4x1x4xf16>
  util.return %0 : tensor<4x4x1x4xf16>
}
// RESHAPE-LABEL: util.func public @scatter_batch_and_slice_dims
//  RESHAPE-SAME:   %[[UPDATE:[a-zA-Z0-9]+]]: tensor<4x1x1x4xf16>
//  RESHAPE-SAME:   %[[INDICES:.+]]: tensor<4x1x2xi64>
//       RESHAPE:   %[[ORIGINAL:.+]] = tensor.empty() : tensor<4x4x1x4xf16>
//       RESHAPE:   %[[INDICES_COLLAPSE:.+]] = tensor.collapse_shape %[[INDICES]]
//  RESHAPE-SAME:     tensor<4x1x2xi64> into tensor<4x2xi64>
//       RESHAPE:   %[[ORIGINAL_COLLAPSE:.+]] = tensor.collapse_shape %[[ORIGINAL]]
//  RESHAPE-SAME:     tensor<4x4x1x4xf16> into tensor<4x4x4xf16>
//       RESHAPE:   %[[UPDATE_COLLAPSE:.+]] = tensor.collapse_shape %[[UPDATE]]
//  RESHAPE-SAME:     tensor<4x1x1x4xf16> into tensor<4x4xf16>
//       RESHAPE:   iree_linalg_ext.scatter
//  RESHAPE-SAME:     ins(%[[UPDATE_COLLAPSE]], %[[INDICES_COLLAPSE]]
//  RESHAPE-SAME:     outs(%[[ORIGINAL_COLLAPSE]]

// SLICE-LABEL: util.func public @scatter_batch_and_slice_dims
//  SLICE-SAME:   %[[UPDATE:[a-zA-Z0-9]+]]: tensor<4x1x1x4xf16>
//  SLICE-SAME:   %[[INDICES:.+]]: tensor<4x1x2xi64>
//       SLICE:   %[[ORIGINAL:.+]] = tensor.empty() : tensor<4x4x1x4xf16>
//       SLICE:   %[[INDICES_SLICE:.+]] = tensor.extract_slice %[[INDICES]]
//  SLICE-SAME:     tensor<4x1x2xi64> to tensor<4x2xi64>
//       SLICE:   %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATE]]
//  SLICE-SAME:     tensor<4x1x1x4xf16> to tensor<4x1x4xf16>
//       SLICE:   %[[ORIGINAL_SLICE:.+]] = tensor.extract_slice %[[ORIGINAL]]
//  SLICE-SAME:     tensor<4x4x1x4xf16> to tensor<4x4x4xf16>
//       SLICE:   %[[UPDATE_SLICE2:.+]] = tensor.extract_slice %[[UPDATE_SLICE]]
//  SLICE-SAME:     tensor<4x1x4xf16> to tensor<4x4xf16>
//       SLICE:   iree_linalg_ext.scatter
//  SLICE-SAME:     ins(%[[UPDATE_SLICE2]], %[[INDICES_SLICE]]
//  SLICE-SAME:     outs(%[[ORIGINAL_SLICE]]

// -----

util.func public @scatter_no_change_output(%slice: tensor<1x2xf16>, %indices: tensor<1x2x2xi64>) -> tensor<2x2xf16> {
  %empty = tensor.empty() : tensor<2x2xf16>
  %0 = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
          ins(%slice, %indices: tensor<1x2xf16>, tensor<1x2x2xi64>)
          outs(%empty: tensor<2x2xf16>){
  ^bb0(%in : f16, %out : f16):
    iree_linalg_ext.yield %in : f16
  }-> tensor<2x2xf16>
  util.return %0 : tensor<2x2xf16>
}
// RESHAPE-LABEL: util.func public @scatter_no_change_output
//  RESHAPE-SAME:   %[[UPDATE:[a-zA-Z0-9]+]]: tensor<1x2xf16>
//  RESHAPE-SAME:   %[[INDICES:.+]]: tensor<1x2x2xi64>
//       RESHAPE:   %[[ORIGINAL:.+]] = tensor.empty() : tensor<2x2xf16>
//       RESHAPE:   %[[INDICES_COLLAPSE:.+]] = tensor.collapse_shape %[[INDICES]]
//  RESHAPE-SAME:     tensor<1x2x2xi64> into tensor<2x2xi64>
//       RESHAPE:   %[[UPDATE_COLLAPSE:.+]] = tensor.collapse_shape %[[UPDATE]]
//  RESHAPE-SAME:     tensor<1x2xf16> into tensor<2xf16>
//       RESHAPE:   iree_linalg_ext.scatter
//  RESHAPE-SAME:     ins(%[[UPDATE_COLLAPSE]], %[[INDICES_COLLAPSE]]
//  RESHAPE-SAME:     outs(%[[ORIGINAL]]

// SLICE-LABEL: util.func public @scatter_no_change_output
//  SLICE-SAME:   %[[UPDATE:[a-zA-Z0-9]+]]: tensor<1x2xf16>
//  SLICE-SAME:   %[[INDICES:.+]]: tensor<1x2x2xi64>
//       SLICE:   %[[ORIGINAL:.+]] = tensor.empty() : tensor<2x2xf16>
//       SLICE:   %[[INDICES_SLICE:.+]] = tensor.extract_slice %[[INDICES]]
//  SLICE-SAME:     tensor<1x2x2xi64> to tensor<2x2xi64>
//       SLICE:   %[[UPDATE_SLICE:.+]] = tensor.extract_slice %[[UPDATE]]
//  SLICE-SAME:     tensor<1x2xf16> to tensor<2xf16>
//       SLICE:   iree_linalg_ext.scatter
//  SLICE-SAME:     ins(%[[UPDATE_SLICE]], %[[INDICES_SLICE]]
//  SLICE-SAME:     outs(%[[ORIGINAL]]

// -----

util.func public @map_scatter(%input: tensor<1x2x1x1xf16>) -> tensor<2x2x2x2xf16> {
  %empty = tensor.empty() : tensor<2x2x2x2xf16>
  %0 = iree_linalg_ext.map_scatter %input into %empty {
    ^bb0(%idx0: index, %idx1: index, %idx2: index, %idx3: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %idx2, %idx3, %mask : index, index, index, index, i1
  } : tensor<1x2x1x1xf16> into tensor<2x2x2x2xf16> -> tensor<2x2x2x2xf16>
  util.return %0 : tensor<2x2x2x2xf16>
}
// RESHAPE-LABEL: util.func public @map_scatter
//  RESHAPE-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x2x1x1xf16>
//   RESHAPE-DAG:   %[[DEST:.+]] = tensor.empty() : tensor<2x2x2x2xf16>
//   RESHAPE-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   RESHAPE-DAG:   %[[INPUT_COLLAPSE:.+]] = tensor.collapse_shape %[[INPUT]]
//  RESHAPE-SAME:     tensor<1x2x1x1xf16> into tensor<2xf16>
//       RESHAPE:   iree_linalg_ext.map_scatter %[[INPUT_COLLAPSE]] into %[[DEST]]
//       RESHAPE:     ^bb0(%[[IDX:.+]]: index):
//       RESHAPE:       iree_linalg_ext.yield  %[[C0]], %[[IDX]], %[[C0]], %[[C0]]

// SLICE-LABEL: util.func public @map_scatter
//  SLICE-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x2x1x1xf16>
//   SLICE-DAG:   %[[DEST:.+]] = tensor.empty() : tensor<2x2x2x2xf16>
//   SLICE-DAG:   %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[INPUT]]
//  SLICE-SAME:     tensor<1x2x1x1xf16> to tensor<2xf16>
//       SLICE:   iree_linalg_ext.map_scatter %[[INPUT_SLICE]] into %[[DEST]]
//       SLICE:     ^bb0(%[[IDX:.+]]: index):
//       SLICE:       %[[C0:.+]] = arith.constant 0 : index
//       SLICE:       iree_linalg_ext.yield  %[[C0]], %[[IDX]], %[[C0]], %[[C0]]

// -----

util.func public @map_scatter_all_unit(%input: tensor<1x1xf16>) -> tensor<2x2xf16> {
  %empty = tensor.empty() : tensor<2x2xf16>
  %0 = iree_linalg_ext.map_scatter %input into %empty {
    ^bb0(%idx0: index, %idx1: index):
      %mask = arith.constant true
      iree_linalg_ext.yield %idx0, %idx1, %mask : index, index, i1
  } : tensor<1x1xf16> into tensor<2x2xf16> -> tensor<2x2xf16>
  util.return %0 : tensor<2x2xf16>
}
// RESHAPE-LABEL: util.func public @map_scatter_all_unit
//  RESHAPE-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x1xf16>
//   RESHAPE-DAG:   %[[DEST:.+]] = tensor.empty() : tensor<2x2xf16>
//   RESHAPE-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   RESHAPE-DAG:   %[[INPUT_COLLAPSE:.+]] = tensor.collapse_shape %[[INPUT]]
//  RESHAPE-SAME:     tensor<1x1xf16> into tensor<1xf16>
//       RESHAPE:   iree_linalg_ext.map_scatter %[[INPUT_COLLAPSE]] into %[[DEST]]
//       RESHAPE:       iree_linalg_ext.yield  %[[C0]], %[[C0]]

// SLICE-LABEL: util.func public @map_scatter_all_unit
//  SLICE-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x1xf16>
//   SLICE-DAG:   %[[DEST:.+]] = tensor.empty() : tensor<2x2xf16>
//   SLICE-DAG:   %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[INPUT]]
//  SLICE-SAME:     tensor<1x1xf16> to tensor<1xf16>
//       SLICE:   iree_linalg_ext.map_scatter %[[INPUT_SLICE]] into %[[DEST]]
//       SLICE:       %[[C0:.+]] = arith.constant 0 : index
//       SLICE:       iree_linalg_ext.yield  %[[C0]], %[[C0]]
