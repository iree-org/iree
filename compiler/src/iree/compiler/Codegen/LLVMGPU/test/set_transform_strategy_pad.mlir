// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN:   --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy \
// RUN: | FileCheck %s

// Check that setting the command line options affect the transform
// strategy as expected.
// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN:   --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy \
// RUN:   --td-pad-strategy-blk-sizes=16,32,1 \
// RUN:   --td-pad-strategy-num-threads=8,4,1 \
// RUN:   --td-pad-strategy-vector-size=2,4 \
// RUN:   --td-pad-strategy-use-async-copies=false \
// RUN: | FileCheck --check-prefix=WITH_OPTIONS %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.abbr_target<cuda:"sm_80">}>
module {
  func.func @pad() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
    %c0 = arith.constant 0 : index
    %c56 = arith.constant 56 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<123x456xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [123, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x456xf32>> -> tensor<123x456xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %2 low[%c0, 0] high[5, %c56] {
    ^bb0(%arg0: index, %arg1: index):
      tensor.yield %cst_0 : f32
    } : tensor<123x456xf32> to tensor<128x512xf32>
    flow.dispatch.tensor.store %padded, %1, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    return
  }
}

// CHECK-LABEL: func @pad
//       CHECK:   transform.named_sequence
//       CHECK:   transform.iree.register_match_callbacks
//       CHECK:   {{.*}} = transform.iree.match_callback failures(propagate) "pad"({{.*}}) : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.structured.tile_using_forall {{.*}}   tile_sizes [64, 64](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//       CHECK:   apply_patterns to %{{.*}} {
//       CHECK:     transform.apply_patterns.canonicalization
//       CHECK    }
//       CHECK:   transform.iree.apply_licm
//       CHECK:   transform.apply_cse to
//       CHECK:   {{.*}} = transform.structured.match ops{["scf.if"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.scf.take_assumed_branch {{.*}} take_else_branch : (!transform.any_op) -> ()
//       CHECK:   transform.iree.populate_workgroup_count_region_using_num_threads_slice {{.*}} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.tile_using_forall {{.*}}   num_threads [16, 16](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//       CHECK:   apply_patterns to %{{.*}} {
//       CHECK:     transform.apply_patterns.canonicalization
//       CHECK    }
//       CHECK:   transform.iree.apply_licm
//       CHECK:   transform.apply_cse to
//       CHECK:   {{.*}} = transform.structured.match ops{["scf.if"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.scf.take_assumed_branch {{.*}} take_else_branch : (!transform.any_op) -> ()
//       CHECK:   transform.structured.vectorize {{.*}} vector_sizes [4, 4] : !transform.any_op
//       CHECK:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:     transform.apply_patterns.vector.lower_masked_transfers
//       CHECK:   apply_patterns to %{{.*}} {
//   CHECK-DAG:     transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
//   CHECK-DAG:     transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
//   CHECK-DAG:     transform.apply_patterns.vector.cast_away_vector_leading_one_dim
//       CHECK:   } : !transform.any_op
//       CHECK:   {{.*}} = transform.structured.vectorize_children_and_apply_patterns {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   apply_patterns to %{{.*}} {
//       CHECK:     transform.apply_patterns.canonicalization
//       CHECK    }
//       CHECK:   transform.iree.apply_licm
//       CHECK:   transform.apply_cse to
//       CHECK:   transform.iree.eliminate_empty_tensors {{.*}} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.iree.bufferize {target_gpu} {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.memref.erase_dead_alloc_and_stores {{.*}} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.iree.forall_to_workgroup {{.*}} : (!transform.any_op) -> ()
//       CHECK:   transform.iree.map_nested_forall_to_gpu_threads {{.*}} workgroup_dims = [16, 16, 1] subgroup_size = 32 sync_after_distribution = true : (!transform.any_op) -> ()
//       CHECK:     transform.apply_patterns.vector.lower_masks
//       CHECK:     transform.apply_patterns.vector.materialize_masks
//       CHECK:   apply_patterns to %{{.*}} {
//   CHECK-DAG:     transform.apply_patterns.linalg.tiling_canonicalization
//   CHECK-DAG:     transform.apply_patterns.memref.fold_memref_alias_ops
//   CHECK-DAG:     transform.apply_patterns.canonicalization
//       CHECK:   } : !transform.any_op
//       CHECK:   transform.iree.apply_licm
//       CHECK:   transform.apply_cse to

// WITH_OPTIONS-LABEL: func @pad
//       WITH_OPTIONS:   transform.structured.tile_using_forall {{.*}}   tile_sizes [32, 16](mapping = [#gpu.block<y>, #gpu.block<x>])
//       WITH_OPTIONS:   {{.*}} = transform.structured.tile_using_forall {{.*}}   num_threads [4, 8](mapping = [#gpu.thread<y>, #gpu.thread<x>])
//       WITH_OPTIONS:   transform.structured.vectorize {{.*}} vector_sizes [2, 4] : !transform.any_op
//       WITH_OPTIONS:   transform.iree.map_nested_forall_to_gpu_threads {{.*}} workgroup_dims = [8, 4, 1]

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.abbr_target<cuda:"sm_80">}>
module {
  func.func @pad_low() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<123x456xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [123, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x456xf32>> -> tensor<123x456xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %2 low[5, 0] high[0, 56] {
    ^bb0(%arg0: index, %arg1: index):
      tensor.yield %cst_0 : f32
    } : tensor<123x456xf32> to tensor<128x512xf32>
    flow.dispatch.tensor.store %padded, %1, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    return
  }
}

// The strategy doesn't apply for low padding.
// CHECK-LABEL: @pad_low
// CHECK-NOT: transform.iree
// WITH_OPTIONS-LABEL: @pad_low
// WITH_OPTIONS-NOT: transform.iree

// -----

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.abbr_target<cuda:"sm_80">}>
module {
  func.func @pad_local() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<123x456xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [123, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x456xf32>> -> tensor<123x456xf32>
    %padded = tensor.pad %2 low[0, 0] high[5, 56] {
    ^bb0(%arg0: index, %arg1: index):
      %3 = arith.index_cast %arg0 : index to i64
      %4 = arith.uitofp %3 : i64 to f32
      tensor.yield %4 : f32
    } : tensor<123x456xf32> to tensor<128x512xf32>
    flow.dispatch.tensor.store %padded, %1, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    return
  }
}

// The strategy doesn't apply for local pad values.
// CHECK-LABEL: @pad_local
// CHECK-NOT: transform.iree
// WITH_OPTIONS-LABEL: @pad_local
// WITH_OPTIONS-NOT: transform.iree
