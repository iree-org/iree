// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches, cse))" --split-input-file --mlir-print-local-scope %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches, iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-dispatch-creation-materialize-default-workgroup-count-region, canonicalize))" --split-input-file --mlir-print-local-scope %s | FileCheck %s --check-prefix=WORKGROUP
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches{enable-fuse-pad}, cse))" --split-input-file --mlir-print-local-scope %s | FileCheck %s --check-prefix=FUSE-PAD

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
// CHECK-LABEL:  @split_reduction_dynamic
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
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
//       CHECK:      } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
//       CHECK:      flow.return %[[FORALL]] : tensor<?x?xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:        ins(%[[DISPATCH]] :
//  CHECK-SAME:        outs(%[[FILL0]] :
//  CHECK-SAME:        dimensions = [1]
//       CHECK:    util.return %[[REDUCE]]

// Check that count region contains splitk modifier.
// WORKGROUP-LABEL:   @split_reduction_dynamic(
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
// CHECK-LABEL:  @split_reduction_static(
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<64x4096xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall
//       CHECK:        linalg.generic
//       CHECK:      flow.return %[[FORALL]] : tensor<64x32xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:        ins(%[[DISPATCH]] :
//  CHECK-SAME:        dimensions = [1]
//       CHECK:    util.return %[[REDUCE]]

// Check that count region contains splitk modifier.
// WORKGROUP-LABEL:   @split_reduction_static
//       WORKGROUP:     count(
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_from_slice()
//       WORKGROUP:       iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier

// -----

// Test multiple reduction dimensions.

util.func public @split_reduction_multiple_dims(%arg0: tensor<?x?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = ["parallel", "reduction", "reduction"],
      iree_linalg_ext.split_reduction = [128, 16]}
      ins(%arg0 : tensor<?x?x?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}
// CHECK-LABEL:  @split_reduction_multiple_dims(
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:    %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:    %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//   CHECK-DAG:    %[[PARTIAL_D1:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%[[D1]]]
//   CHECK-DAG:    %[[PARTIAL_D2:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%[[D2]]]
//       CHECK:    %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[PARTIAL_D1]], %[[PARTIAL_D2]]) : tensor<?x?x?xf32>
//       CHECK:    %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:      %[[FORALL:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
//  CHECK-SAME:          (0, 0) to (%[[D1]], %[[D2]]) step (128, 16)
//  CHECK-SAME:          shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<?x?x?xf32>) {
//   CHECK-DAG:        %[[T1:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%[[IV0]])[%[[D1]]]
//   CHECK-DAG:        %[[T2:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 16)>(%[[IV1]])[%[[D2]]]
//       CHECK:        %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV0]], %[[IV1]]] [%[[D0]], %[[T1]], %[[T2]]]
//   CHECK-DAG:        %[[OUT_OFFSET0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 128)>()[%[[IV0]]]
//   CHECK-DAG:        %[[OUT_OFFSET1:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%[[IV1]]]
//       CHECK:        %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][0, %[[OUT_OFFSET0]], %[[OUT_OFFSET1]]] [%[[D0]], 1, 1]
//       CHECK:        %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:            outs(%[[INIT_SLICE]] :
//       CHECK:        %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:            iterator_types = ["parallel", "reduction", "reduction"]
//  CHECK-SAME:            ins(%[[INPUT_SLICE]] :
//  CHECK-SAME:            outs(%[[FILL]] :
//       CHECK:        tensor.parallel_insert_slice %[[GENERIC]] into %[[INIT]]
//       CHECK:      } {mapping = [#iree_linalg_ext.split_reduction_mapping<1>, #iree_linalg_ext.split_reduction_mapping<0>]}
//       CHECK:      flow.return %[[FORALL]] : tensor<?x?x?xf32>
//       CHECK:    %[[REDUCE:.+]] = linalg.reduce
//  CHECK-SAME:        ins(%[[DISPATCH]] :
//  CHECK-SAME:        dimensions = [1, 2]
//       CHECK:    util.return %[[REDUCE]]

// -----

util.func public @fuse_pad_with_conv(%arg0 : tensor<16x225x225x16xbf16>, %arg1 : tensor<16x225x225x64xbf16>) -> (tensor<64x3x3x16xf32>) {
  %cst = arith.constant 0.0 : bf16
  %cst_0 = arith.constant 0.0 : f32
  %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
    tensor.yield %cst : bf16
  } : tensor<16x225x225x16xbf16> to tensor<16x227x227x16xbf16>
  %2 = tensor.empty() : tensor<64x3x3x16xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x3x3x16xf32>) -> tensor<64x3x3x16xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<16x227x227x16xbf16>, tensor<16x225x225x64xbf16>) outs(%3 : tensor<64x3x3x16xf32>) attrs =  {iree_linalg_ext.split_reduction = [1, 255, 255]} {
  ^bb0(%in: bf16, %in_1: bf16, %out: f32):
    %9 = arith.extf %in : bf16 to f32
    %10 = arith.extf %in_1 : bf16 to f32
    %11 = arith.mulf %9, %10 : f32
    %12 = arith.addf %out, %11 : f32
    linalg.yield %12 : f32
  } -> tensor<64x3x3x16xf32>
  util.return %4 : tensor<64x3x3x16xf32>
}
// FUSE-PAD-LABEL:  @fuse_pad_with_conv(
//       FUSE-PAD:    %[[DISPATCH:.+]] = flow.dispatch.region
//       FUSE-PAD:      scf.forall
//   FUSE-PAD-NOT:        scf.if
//       FUSE-PAD:        tensor.extract_slice
//       FUSE-PAD:        tensor.pad
//       FUSE-PAD:        linalg.generic
//       FUSE-PAD:    linalg.reduce
//       FUSE-PAD:      ins(%[[DISPATCH]]

// Not enabled: pad should be outside of dispatch.
// CHECK-LABEL:  @fuse_pad_with_conv(
//       CHECK:    tensor.pad
//       CHECK:    flow.dispatch.region
//       CHECK:      linalg.generic
