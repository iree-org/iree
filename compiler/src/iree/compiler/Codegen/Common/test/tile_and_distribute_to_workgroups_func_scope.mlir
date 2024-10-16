// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups{distribution-method=2}, canonicalize, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

func.func @multiple_dim_distribute(%s0 : index, %s1 : index, %s2 : index, %s3 : index,
    %arg0 : tensor<2x3x4x5xf32>) attributes {
    translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>} {
  %c0 = arith.constant 0 : index
  %result = hal.interface.binding.subspan layout(
      <bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
                   #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>)
      binding(0) alignment(64) offset(%c0) flags(Indirect)
      : !flow.dispatch.tensor<writeonly:tensor<?x2x?x3x?x4x?x5xf32>>{%s0, %s1, %s2, %s3}
  %35 = tensor.empty(%s0, %s1, %s2, %s3) : tensor<?x2x?x3x?x4x?x5xf32>
  %36 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d3, d5, d7)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<2x3x4x5xf32>) outs(%35 : tensor<?x2x?x3x?x4x?x5xf32>)
      attrs =  {lowering_config = #iree_gpu.lowering_config<{thread = [1, 1, 1, 1, 1, 1, 1, 1], workgroup = [1, 2, 1, 4, 1, 4, 1, 1]}>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<?x2x?x3x?x4x?x5xf32>
  flow.dispatch.tensor.store %36, %result, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [%s0, 2, %s1, 3, %s2, 4, %s3, 5], strides = [1, 1, 1, 1, 1, 1, 1, 1]
      : tensor<?x2x?x3x?x4x?x5xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x2x?x3x?x4x?x5xf32>>{%s0, %s1, %s2, %s3}
  return
}
// CHECK-LABEL: func @multiple_dim_distribute(
//  CHECK-SAME:     %[[S0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[S1:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[S2:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[S3:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[INPUT:.+]]: tensor<2x3x4x5xf32>)
//   CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//   CHECK-DAG:   %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x2x1x3x1x4x1x1xf32>
//   CHECK-DAG:   %[[IN_SLICE:.+]] = tensor.extract_slice %[[INPUT]][0, 0, 0, %[[WG_ID_X]]] [2, 3, 4, 1]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[IN_SLICE]] :
//  CHECK-SAME:       outs(%[[EMPTY]] :
//   CHECK-DAG:   %[[WG_ID_Z_0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s1 floordiv s2) floordiv s0)>()[%[[S1]], %[[WG_ID_Z]], %[[S2]]]
//   CHECK-DAG:   %[[WG_ID_Z_1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s1 floordiv s2) mod s0)>()[%[[S1]], %[[WG_ID_Z]], %[[S2]]]
//   CHECK-DAG:   %[[WG_ID_Z_2:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 mod s1)>()[%[[WG_ID_Z]], %[[S2]]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]],
//  CHECK-SAME:       offsets = [%[[WG_ID_Z_0]], 0, %[[WG_ID_Z_1]], 0, %[[WG_ID_Z_2]], 0, %[[WG_ID_Y]], %[[WG_ID_X]]]
//  CHECK-SAME:       sizes = [1, 2, 1, 3, 1, 4, 1, 1]
