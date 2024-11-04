// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

func.func @custom_op(%arg0 : tensor<384x512xf32>, %arg1 : tensor<512x128xf32>,
    %arg2 : tensor<128xf32>) -> tensor<384x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<384x128xf32>
  %1 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d1)>,
                       affine_map<(d0, d1)[s0] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1, %arg2 : tensor<384x512xf32>, tensor<512x128xf32>, tensor<128xf32>)
      outs(%0 : tensor<384x128xf32>) {
    ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?xf32>, %t3 : tensor<?x?xf32>):
      %2 = linalg.fill ins(%cst : f32) outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %3 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %4 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %t2 : tensor<?x?xf32>, tensor<?xf32>)
          outs(%t3 : tensor<?x?xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %5 = arith.addf %b0, %b1 : f32
          linalg.yield %5 : f32
      } -> tensor<?x?xf32>
      iree_linalg_ext.yield %4 : tensor<?x?xf32>
  } -> tensor<384x128xf32>
  return %1 : tensor<384x128xf32>
}
//      CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0]]>
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64,
//      CHECK: func @custom_op
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.custom_op
// CHECK-SAME:       lowering_config = #[[CONFIG]]
//      CHECK:   ^bb
//      CHECK:     linalg.matmul
// CHECK-SAME:         lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, promote_operands = [0, 1], reduction = [0, 0, 32], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [64, 64, 0]}>
//      CHECK:   iree_linalg_ext.yield

// -----

func.func @custom_op_preset_config(%arg0: tensor<384x512xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<128xf32>) -> tensor<384x128xf32>
  attributes {translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<384x128xf32>
  %1 = iree_linalg_ext.custom_op{
      indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d1)>,
                       affine_map<(d0, d1)[s0] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      attributes {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[24, 32]]>}
      ins(%arg0, %arg1, %arg2 : tensor<384x512xf32>, tensor<512x128xf32>, tensor<128xf32>) outs(%0 : tensor<384x128xf32>) {
  ^bb0(%arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: tensor<?xf32>, %arg6: tensor<?x?xf32>):
    %2 = linalg.fill ins(%cst : f32) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = linalg.matmul ins(%arg3, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%3, %arg5 : tensor<?x?xf32>, tensor<?xf32>) outs(%arg6 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
    iree_linalg_ext.yield %4 : tensor<?x?xf32>
  } -> tensor<384x128xf32>
  return %1 : tensor<384x128xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[24, 32]]>
//  CHECK-DAG: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<LLVMGPUTileAndFuse>
//      CHECK: func @custom_op_preset_config(
// CHECK-SAME:     translation_info = #[[TRANSLATION_INFO]]
//      CHECK:   iree_linalg_ext.custom_op
// CHECK-SAME:       lowering_config = #[[CONFIG]]
//  CHECK-NOT:   lowering_config
