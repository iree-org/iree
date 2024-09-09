// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @tensor_cast_to_reshape(%reshape_17 : tensor<?x?x?x?xf32>, %65 : tensor<?x12x?x64xf32>, %0 : index, %1 : index) -> tensor<?x?x?x?xf32> {
  %cast = tensor.cast %reshape_17 : tensor<?x?x?x?xf32> to tensor<?x?x12x64xf32>
  %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%cast : tensor<?x?x12x64xf32>) outs(%65 : tensor<?x12x?x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x12x?x64xf32>
  %cast_18 = tensor.cast %66 : tensor<?x12x?x64xf32> to tensor<?x?x?x?xf32>
  util.return  %cast_18 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: util.func public @tensor_cast_to_reshape
//       CHECK:   flow.tensor.reshape
//       CHECK-SAME: tensor<?x?x?x?xf32>
//       CHECK-SAME: -> tensor<?x?x12x64xf32>
//       CHECK:   linalg.generic
//       CHECK:   flow.tensor.reshape
//       CHECK-SAME: tensor<?x12x?x64xf32>
//       CHECK-SAME: -> tensor<?x?x?x?xf32>
