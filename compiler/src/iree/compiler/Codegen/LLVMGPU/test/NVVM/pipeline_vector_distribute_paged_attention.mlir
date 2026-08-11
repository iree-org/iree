// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-llvmgpu-nvvm-lowering-pipeline='include-llvm-lowering=false' \
// RUN:   %s | FileCheck %s

/// Paged attention reduction distribution to multiple subgroups.
/// Distribute 8x32 reduction dims across 4 subgroups with 2x16 threads shape per subgroup.

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [128, 1, 1]
    subgroup_size = 32>
#qk_attrs_config = #iree_gpu.lowering_config<{
    subgroup_basis = [[1, 1, 1, 1, 1, 4, 1], [4, 3, 2, 1, 5, 6]],
    thread = [0, 0, 0, 8, 0, 0],
    lane_basis = [[1, 1, 1, 1, 1, 2, 16], [2, 1, 0, 4, 5, 6]]
}>
#pv_attrs_config = #iree_gpu.lowering_config<{
    subgroup_basis = [[1, 1, 1, 1, 1, 4, 1], [4, 3, 2, 1, 5, 6]],
    thread = [0, 0, 0, 8, 0, 0],
    lane_basis = [[1, 1, 1, 1, 1, 2, 16], [2, 1, 0, 4, 5, 6]]
}>
#attention_lowering_config = #iree_gpu.lowering_config<{
    partial_reduction = [0, 0, 0, 0, 0, 8, 0],
    workgroup = [1, 1, 1, 0, 0, 0, 0]
}>
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @paged_attention_4xDx1x32x128_f16() attributes {hal.executable.target = #executable_target_cuda, translation_info = #translation} {
  %scale = arith.constant 8.837890e-02 : f16
  %zero_f16 = arith.constant 0.000000e+00 : f16
  %neg_inf_f16 = arith.constant 0xFC00 : f16
  %zero_f32 = arith.constant 0.000000e+00 : f32
  %neg_inf_f32 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index

  %num_pages_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %num_pages_index = arith.index_castui %num_pages_i32 : i32 to index
  %num_pages = util.assume.int %num_pages_index<umin = 1, umax = 512> : index
  %workload = iree_tensor_ext.dispatch.workload.ordinal %num_pages, 0 : index

  %kv_storage = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1x32x128xf16>>
  %key_page_table = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%workload}
  %value_page_table = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%workload}
  %query = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x1x128xf16>>
  %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x1x128xf32>>

  %key_indices = iree_tensor_ext.dispatch.tensor.load %key_page_table, offsets = [0, 0], sizes = [4, %workload], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%workload} -> tensor<4x?xi64>
  %value_indices = iree_tensor_ext.dispatch.tensor.load %value_page_table, offsets = [0, 0], sizes = [4, %workload], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%workload} -> tensor<4x?xi64>
  %query_tensor = iree_tensor_ext.dispatch.tensor.load %query, offsets = [0, 0, 0, 0], sizes = [4, 1, 1, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x1x128xf16>> -> tensor<4x1x1x128xf16>
  %storage_tensor = iree_tensor_ext.dispatch.tensor.load %kv_storage, offsets = [0, 0, 0, 0], sizes = [4096, 1, 32, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1x32x128xf16>> -> tensor<4096x1x32x128xf16>

  %output_empty = tensor.empty() : tensor<4x1x1x128xf32>
  %mask_empty = tensor.empty(%workload) : tensor<4x1x1x?x32xf16>
  %gather_empty = tensor.empty(%workload) : tensor<4x?x1x32x128xf16>
  %reduction_empty = tensor.empty() : tensor<4x1x1xf32>
  %output_init = linalg.fill ins(%zero_f32 : f32) outs(%output_empty : tensor<4x1x1x128xf32>) -> tensor<4x1x1x128xf32>
  %max_init = linalg.fill ins(%neg_inf_f32 : f32) outs(%reduction_empty : tensor<4x1x1xf32>) -> tensor<4x1x1xf32>
  %sum_init = linalg.fill ins(%zero_f32 : f32) outs(%reduction_empty : tensor<4x1x1xf32>) -> tensor<4x1x1xf32>

  %key = iree_linalg_ext.gather dimension_map = [0]
      ins(%storage_tensor, %key_indices : tensor<4096x1x32x128xf16>, tensor<4x?xi64>)
      outs(%gather_empty : tensor<4x?x1x32x128xf16>) -> tensor<4x?x1x32x128xf16>
  %value = iree_linalg_ext.gather dimension_map = [0]
      ins(%storage_tensor, %value_indices : tensor<4096x1x32x128xf16>, tensor<4x?xi64>)
      outs(%gather_empty : tensor<4x?x1x32x128xf16>) -> tensor<4x?x1x32x128xf16>

  %mask = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    }
    outs(%mask_empty : tensor<4x1x1x?x32xf16>) {
      ^bb0(%out: f16):
        %position_in_page = linalg.index 4 : index
        %page_index = linalg.index 3 : index
        %page_start = arith.muli %page_index, %c32 overflow<nsw> : index
        %kv_position = arith.addi %page_start, %position_in_page : index
        %is_causal = arith.cmpi sle, %kv_position, %c0 : index
        %mask_value = arith.select %is_causal, %zero_f16, %neg_inf_f16 : f16
        linalg.yield %mask_value : f16
    } -> tensor<4x1x1x?x32xf16>

  %attention:3 = iree_linalg_ext.online_attention {
      decomposition_config = {
        pv_attrs = {lowering_config = #pv_attrs_config},
        qk_attrs = {lowering_config = #qk_attrs_config}
      },
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>],
      lowering_config = #attention_lowering_config
    }
    ins(%query_tensor, %key, %value, %scale, %mask : tensor<4x1x1x128xf16>, tensor<4x?x1x32x128xf16>, tensor<4x?x1x32x128xf16>, f16, tensor<4x1x1x?x32xf16>)
    outs(%output_init, %max_init, %sum_init : tensor<4x1x1x128xf32>, tensor<4x1x1xf32>, tensor<4x1x1xf32>) {
      ^bb0(%score: f32):
        iree_linalg_ext.yield %score : f32
    } -> tensor<4x1x1x128xf32>, tensor<4x1x1xf32>, tensor<4x1x1xf32>

  iree_tensor_ext.dispatch.tensor.store %attention#0, %output, offsets = [0, 0, 0, 0], sizes = [4, 1, 1, 128], strides = [1, 1, 1, 1] : tensor<4x1x1x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x1x128xf32>>
  return
}

// CHECK-LABEL: func.func @paged_attention_4xDx1x32x128_f16
// CHECK-NOT: iree_linalg_ext.gather
// CHECK-NOT: iree_linalg_ext.online_attention
// CHECK: scf.for
// CHECK: gpu.subgroup_reduce
// CHECK: iree_codegen.dispatch_config @paged_attention_4xDx1x32x128_f16 workgroup_size = [128, 1, 1] subgroup_size = 32
