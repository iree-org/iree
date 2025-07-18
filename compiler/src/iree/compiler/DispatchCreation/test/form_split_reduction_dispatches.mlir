// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches, cse))" --split-input-file --mlir-print-local-scope %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches, iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-dispatch-creation-materialize-default-workgroup-count-region, canonicalize))" --split-input-file --mlir-print-local-scope %s | FileCheck %s --check-prefix=WORKGROUP

util.func public @split_reduction_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"],
      iree_linalg_ext.split_reduction = [128]}
      ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}
// CHECK-LABEL:  util.func public @split_reduction_dynamic
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:    %[[EMPTY0:.+]] = tensor.empty(%{{.+}}) : tensor<?xf32>
//       CHECK:    %[[FILL0:.+]] = linalg.fill
//  CHECK-SAME:        outs(%[[EMPTY0]] :
//   CHECK-DAG:    %[[EMPTY1:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall
//  CHECK-SAME:          shared_outs(%[[INIT:.+]] = %[[EMPTY1]]) -> (tensor<?x?xf32>) {
//       CHECK:        %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//       CHECK:        %[[FILL1:.+]] = linalg.fill
//  CHECK-SAME:            outs(%[[INIT_SLICE]] :
//       CHECK:        %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:            outs(%[[FILL1]] :
//       CHECK:        tensor.parallel_insert_slice %[[GENERIC]] into %[[INIT]]
//       CHECK:      } {mapping = [#iree_linalg_ext.split_reduction_mapping]}
//       CHECK:      flow.return %[[FORALL]] : tensor<?x?xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg
//  CHECK-SAME:      ins(%[[DISPATCH]] :
//  CHECK-SAME:      outs(%[[FILL0]] :
//       CHECK:    util.return %[[REDUCE]]

// Check that count region contains splitk modifier.
// WORKGROUP-LABEL:   util.func public @split_reduction_dynamic
//       WORKGROUP:     count(
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_from_slice
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier

// -----

util.func public @split_reduction_static(%arg0: tensor<64x4096xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"],
      iree_linalg_ext.split_reduction = [128]}
      ins(%arg0 : tensor<64x4096xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<64xf32>
  util.return %2 : tensor<64xf32>
}
// CHECK-LABEL:  util.func public @split_reduction_static
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<64x4096xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall
//       CHECK:        linalg.generic
//       CHECK:      flow.return %[[FORALL]] : tensor<64x32xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg
//  CHECK-SAME:      ins(%[[DISPATCH]] :
//       CHECK:    util.return %[[REDUCE]]

// Check that count region contains splitk modifier.
// WORKGROUP-LABEL:   util.func public @split_reduction_static
//       WORKGROUP:     count(
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_from_slice
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier
