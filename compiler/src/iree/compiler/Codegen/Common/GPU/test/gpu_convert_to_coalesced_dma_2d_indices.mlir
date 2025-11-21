// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma))" %s | FileCheck %s

// This test demonstrates a 2D indices gather operation that successfully converts
// to iree_gpu.coalesced_gather_dma with multiple index vectors.
//
// The gather operation uses 2D indices (tensor<4x2xi32>) to index into two dimensions
// of the source tensor. The transformation extracts each column of the indices tensor
// into separate 1D tensors and passes them as variadic operands to coalesced_gather_dma.

#gpu_target_2d = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128]
>>

#exec_target_2d = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_2d}>
#translation_2d = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1024, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>

// This gather operation uses 2D indices to index into two dimensions of the source.
//
// Semantics:
// - source: tensor<64x128x32xf32> - 3D tensor to gather from
// - indices: tensor<4x2xi32> - 4 batch elements, each with 2 index values [i0, i1]
// - dimension_map: [0, 1] - indices[batch, 0] indexes into source dim 0,
//                            indices[batch, 1] indexes into source dim 1
// - Result: output[batch, :] = source[indices[batch, 0], indices[batch, 1], :]
//
// The transformation will extract the 2D indices into two separate tensors:
// - tensor<Nxi32> for dimension 0 indices
// - tensor<Nxi32> for dimension 1 indices

// CHECK-LABEL: func.func @gather_2d_indices_input
func.func @gather_2d_indices_input(%source: tensor<64x128x32xf32>, %indices: tensor<4x2xi32>, %init: tensor<4x32xf32>) -> tensor<4x32xf32>
  attributes {hal.executable.target = #exec_target_2d, translation_info = #translation_2d} {

  // CHECK: scf.forall
  // CHECK:   scf.forall
  // CHECK:     %[[SLICE1:.+]] = tensor.extract_slice %{{.+}}[0, 0] [1, 1] [1, 1] : tensor<1x2xi32> to tensor<1x1xi32>
  // CHECK:     %[[COLLAPSED1:.+]] = tensor.collapse_shape %[[SLICE1]]
  // CHECK:     %[[SLICE2:.+]] = tensor.extract_slice %{{.+}}[0, 1] [1, 1] [1, 1] : tensor<1x2xi32> to tensor<1x1xi32>
  // CHECK:     %[[COLLAPSED2:.+]] = tensor.collapse_shape %[[SLICE2]]
  // CHECK:     iree_gpu.coalesced_gather_dma %{{.+}}[%[[COLLAPSED1]], %[[COLLAPSED2]]] into %{{.+}} lane(%{{.+}})
  // CHECK-SAME:   : tensor<64x128x{{.+}}xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1x{{.+}}xf32>, index

  %result = iree_linalg_ext.gather {lowering_config = #iree_gpu.use_global_load_dma}
    dimension_map = [0, 1]
    ins(%source, %indices : tensor<64x128x32xf32>, tensor<4x2xi32>)
    outs(%init : tensor<4x32xf32>) -> tensor<4x32xf32>

  // CHECK-NOT: iree_linalg_ext.gather
  return %result : tensor<4x32xf32>
}
