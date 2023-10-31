// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" --iree-codegen-llvmgpu-enable-transform-dialect-aligned-matmul | FileCheck %s

// Check that setting the command line options affect the transform
// strategy as expected.
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" \
// RUN: -td-matmul-strategy-blk-sizes=256,64,1 \
// RUN: -td-matmul-strategy-reduc-size=8 \
// RUN: -td-matmul-strategy-num-threads=32,4,1 \
// RUN: -td-matmul-strategy-num-warps=1,4,1 \
// RUN: -td-matmul-strategy-use-async-copies=true \
// RUN: -td-matmul-strategy-use-mma-sync=true \
// RUN: -td-matmul-strategy-pipeline-depth=5 \
// RUN: | FileCheck --check-prefix=WITH_OPTIONS %s

// Check that various more exotic strategies apply properly e2e but without otherwise checking their content.
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" \
// RUN: --iree-codegen-llvmgpu-enable-transform-dialect-aligned-matmul \
// RUN: -td-matmul-strategy-blk-sizes=16,16,1 \
// RUN: -td-matmul-strategy-reduc-size=16 \
// RUN: -td-matmul-strategy-num-threads=32,1,1 \
// RUN: -td-matmul-strategy-num-warps=1,1,1 \
// RUN: -td-matmul-strategy-use-async-copies=true \
// RUN: -td-matmul-strategy-use-mma-sync=true \
// RUN: -td-matmul-strategy-pipeline-depth=9 \
// RUN: | FileCheck --check-prefix=WITH_OPTIONS_2 %s

// Check that various more exotic strategies apply properly e2e but without otherwise checking their content.
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" \
// RUN: --iree-codegen-llvmgpu-enable-transform-dialect-aligned-matmul \
// RUN: -td-matmul-strategy-blk-sizes=128,64,1 \
// RUN: -td-matmul-strategy-reduc-size=16 \
// RUN: -td-matmul-strategy-num-threads=128,2,1 \
// RUN: -td-matmul-strategy-num-warps=1,8,1 \
// RUN: -td-matmul-strategy-use-async-copies=true \
// RUN: -td-matmul-strategy-use-mma-sync=true \
// RUN: -td-matmul-strategy-pipeline-depth=3 \
// RUN: | FileCheck --check-prefix=WITH_OPTIONS_3 %s

// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy)))" --iree-codegen-llvmgpu-enable-transform-dialect-small-matmul \
// RUN: | FileCheck --check-prefix=SMALL %s

hal.executable @matmul_1 {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @matmul_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_1() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2052x2556xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2052xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2052x2052xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2052, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2052x2556xf32>> -> tensor<2052x2556xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2052], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2052xf32>> -> tensor<2556x2052xf32>
      %5 = tensor.empty() : tensor<2052x2052xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2052x2052xf32>) -> tensor<2052x2052xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2052x2556xf32>, tensor<2556x2052xf32>) outs(%6 : tensor<2052x2052xf32>) -> tensor<2052x2052xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2052, 2052], strides = [1, 1] : tensor<2052x2052xf32> -> !flow.dispatch.tensor<writeonly:tensor<2052x2052xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @matmul_1

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.iree.match_callback failures(propagate) "matmul"
// CHECK: transform.structured.tile_using_forall %{{.*}} tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.iree.populate_workgroup_count_region_using_num_threads_slice
// CHECK: transform.structured.tile_using_for %{{.*}}[0, 0, 16]
// CHECK: transform.structured.pad %{{.*}} {copy_back_op = "none", pack_paddings = [1, 1, 1], pad_to_multiple_of = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK: transform.structured.hoist_pad %{{.}} by 1 loops
// CHECK: transform.structured.insert_slice_to_copy %{{.*}} : (!transform.any_op) -> !transform.any_op
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [32, 4](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:   transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [4, 32](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [4, 32](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.vectorize %{{.*}} vector_sizes [4, 4]
// CHECK: transform.structured.vectorize %{{.*}} vector_sizes [4, 4]
// CHECK: transform.structured.vectorize %{{.*}} vector_sizes [32, 4]
// CHECK: transform.apply_patterns.vector.lower_masked_transfers
// CHECK: transform.structured.vectorize_children_and_apply_patterns %{{.*}}
// CHECK: transform.iree.eliminate_empty_tensors %{{.*}}
// CHECK: transform.iree.bufferize {target_gpu} %{{.*}}
// CHECK: transform.iree.forall_to_workgroup %{{.*}}
// CHECK: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [64, 2, 1]
// CHECK: transform.iree.hoist_static_alloc %{{.*}}
// CHECK: apply_patterns to %{{.*}} {
// CHECK:   transform.apply_patterns.memref.fold_memref_alias_ops
// CHECK: } : !transform.any_op
// CHECK: apply_patterns to %{{.*}} {
// CHECK:   transform.apply_patterns.memref.extract_address_computations
// CHECK: } : !transform.any_op
// CHECK: apply_patterns to %{{.*}} {
// CHECK:   transform.apply_patterns.iree.unroll_vectors_gpu_mma_sync
// CHECK: } : !transform.any_op
// CHECK: transform.structured.match ops{["scf.for"]} in %{{.*}}
// CHECK: transform.iree.synchronize_loop %{{.*}}
// CHECK: transform.structured.hoist_redundant_vector_transfers %{{.*}}
// CHECK: transform.memref.erase_dead_alloc_and_stores %{{.*}}
// CHECK: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_mma_sync}
// CHECK: transform.iree.eliminate_gpu_barriers
// CHECK: apply_patterns to %{{.*}} {
// CHECK:   transform.apply_patterns.memref.fold_memref_alias_ops
// CHECK: } : !transform.any_op
// CHECK: transform.memref.multibuffer %{{.*}} {factor = 3 : i64, skip_analysis}
// CHECK: transform.apply_patterns.vector.transfer_to_scf full_unroll = true
// CHECK: transform.iree.create_async_groups %{{.*}} {use_mma_sync}
// CHECK: transform.iree.pipeline_shared_memory_copies %{{.*}} {depth = 3 : i64, use_mma_sync}
// CHECK: transform.apply_patterns.vector.lower_masks
// CHECK: transform.apply_patterns.vector.materialize_masks
// CHECK: apply_patterns to %{{.*}} {
// CHECK-DAG:   transform.apply_patterns.linalg.tiling_canonicalization
// CHECK-DAG:   transform.apply_patterns.memref.fold_memref_alias_ops
// CHECK-DAG:   transform.apply_patterns.canonicalization
// CHECK: } : !transform.any_op
// CHECK: transform.iree.apply_licm
// CHECK: transform.iree.apply_cse

// WITH_OPTIONS-LABEL: func @matmul_1

// WITH_OPTIONS: transform.sequence  failures(propagate) {
// WITH_OPTIONS: transform.iree.match_callback failures(propagate) "matmul"
// Tile sizes are set by td-matmul-strategy-blk-size-XX.
// WITH_OPTIONS: transform.structured.tile_using_forall %{{.*}} tile_sizes [256, 64](mapping = [#gpu.block<y>, #gpu.block<x>])
// WITH_OPTIONS: transform.structured.fuse_into_containing_op
// WITH_OPTIONS: transform.iree.populate_workgroup_count_region_using_num_threads_slice
// The tiling is affected by td-matmul-strategy-reduc-size: 8.
// WITH_OPTIONS: transform.structured.tile_using_for %{{.*}}[0, 0, 8]
// WITH_OPTIONS: transform.structured.pad %{{.*}} {copy_back_op = "none", pack_paddings = [1, 1, 1], pad_to_multiple_of = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// WITH_OPTIONS: transform.structured.hoist_pad %{{.}} by 1 loops
// WITH_OPTIONS: transform.structured.insert_slice_to_copy %{{.*}} : (!transform.any_op) -> !transform.any_op
// WITH_OPTIONS: transform.structured.tile_using_forall %{{.*}}   num_threads [64, 2](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// WITH_OPTIONS:   transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// WITH_OPTIONS: transform.structured.tile_using_forall %{{.*}}   num_threads [8, 16](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// WITH_OPTIONS: transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// WITH_OPTIONS: transform.structured.tile_using_forall %{{.*}}   num_threads [8, 16](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// WITH_OPTIONS: transform.structured.tile_using_forall %{{.*}}   num_threads [4, 1](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// WITH_OPTIONS: transform.structured.tile_using_forall %{{.*}}   num_threads [4, 1](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// WITH_OPTIONS: transform.structured.vectorize %{{.*}} vector_sizes [4, 4]
// WITH_OPTIONS: transform.structured.vectorize %{{.*}} vector_sizes [1, 4]
// WITH_OPTIONS: transform.structured.vectorize %{{.*}} vector_sizes [32, 4]
// WITH_OPTIONS: transform.apply_patterns.vector.lower_masked_transfers
// WITH_OPTIONS: transform.structured.vectorize_children_and_apply_patterns %{{.*}}
// WITH_OPTIONS: transform.iree.eliminate_empty_tensors %{{.*}}
// WITH_OPTIONS: transform.iree.bufferize {target_gpu} %{{.*}}
// WITH_OPTIONS: transform.iree.forall_to_workgroup %{{.*}}
// The workgroup dimensions are controled by td-matmul-strategy-num-threads-XX.
// The warp dimensions are controled by td-matmul-strategy-num-warps-XX.
// WITH_OPTIONS: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [32, 4, 1]
// WITH_OPTIONS: transform.iree.hoist_static_alloc %{{.*}}
// WITH_OPTIONS: apply_patterns to %{{.*}} {
// WITH_OPTIONS:   transform.apply_patterns.memref.fold_memref_alias_ops
// WITH_OPTIONS: } : !transform.any_op
// WITH_OPTIONS: apply_patterns to %{{.*}} {
// WITH_OPTIONS:   transform.apply_patterns.memref.extract_address_computations
// WITH_OPTIONS: } : !transform.any_op
// The unroll attribute should match td-matmul-use-mma-sync, for true: mma_sync,
// for false:_wmma.
// WITH_OPTIONS: apply_patterns to %{{.*}} {
// WITH_OPTIONS:   transform.apply_patterns.iree.unroll_vectors_gpu_mma_sync
// WITH_OPTIONS: }
// WITH_OPTIONS: transform.structured.match ops{["scf.for"]} in %{{.*}}
// WITH_OPTIONS: transform.iree.synchronize_loop %{{.*}}
// WITH_OPTIONS: transform.structured.hoist_redundant_vector_transfers %{{.*}}
// WITH_OPTIONS: transform.memref.erase_dead_alloc_and_stores %{{.*}}
// The attribute should match td-matmul-use-mma-sync.
// WITH_OPTIONS: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_mma_sync}
// WITH_OPTIONS: transform.iree.eliminate_gpu_barriers
// WITH_OPTIONS: apply_patterns to %{{.*}} {
// WITH_OPTIONS:   transform.apply_patterns.memref.fold_memref_alias_ops
// WITH_OPTIONS: } : !transform.any_op
// The multibuffer pass is only run when we set use-async-copies.
// The factor should match td-matmul-strategy-pipeline-depth: 5.
// WITH_OPTIONS: transform.memref.multibuffer %{{.*}} {factor = 5 : i64, skip_analysis}
// WITH_OPTIONS: transform.apply_patterns.vector.transfer_to_scf full_unroll = true
// The attribute should match td-matmul-use-mma-sync.
// WITH_OPTIONS: transform.iree.create_async_groups %{{.*}} {use_mma_sync}
// The depth should match td-matmul-strategy-pipeline-depth: 5.
// WITH_OPTIONS: transform.iree.pipeline_shared_memory_copies %{{.*}} {depth = 5 : i64, use_mma_sync}
// WITH_OPTIONS: transform.apply_patterns.vector.lower_masks
// WITH_OPTIONS: transform.apply_patterns.vector.materialize_masks
// WITH_OPTIONS: apply_patterns to %{{.*}} {
// WITH_OPTIONS:   transform.apply_patterns.linalg.tiling_canonicalization
// WITH_OPTIONS:   transform.apply_patterns.memref.fold_memref_alias_ops
// WITH_OPTIONS: } : !transform.any_op
// WITH_OPTIONS: apply_patterns to %{{.*}} {
// WITH_OPTIONS:   transform.apply_patterns.canonicalization
// WITH_OPTIONS  }
// WITH_OPTIONS: transform.iree.apply_licm
// WITH_OPTIONS: transform.iree.apply_cse


// WITH_OPTIONS_2-LABEL: func @matmul_1

// WITH_OPTIONS_3-LABEL: func @matmul_1

// -----

hal.executable @matmul_2 {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @matmul_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_2() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2051x2555xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2555x2050xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2051x2050xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2051, 2555], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2051x2555xf32>> -> tensor<2051x2555xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2555, 2051], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2555x2050xf32>> -> tensor<2555x2050xf32>
      %5 = tensor.empty() : tensor<2051x2050xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2051x2050xf32>) -> tensor<2051x2050xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2051x2555xf32>, tensor<2555x2050xf32>) outs(%6 : tensor<2051x2050xf32>) -> tensor<2051x2050xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2051, 2050], strides = [1, 1] : tensor<2051x2050xf32> -> !flow.dispatch.tensor<writeonly:tensor<2051x2050xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @matmul_2

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.iree.match_callback failures(propagate) "matmul"
// CHECK: transform.structured.tile_using_forall %{{.*}} tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
// CHECK: transform.iree.populate_workgroup_count_region_using_num_threads_slice
// CHECK: transform.structured.tile_using_for %{{.*}}[0, 0, 16]
// align1
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [8, 16](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// align2
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [2, 64](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// align2
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [2, 64](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// align1
// CHECK: transform.structured.vectorize %{{.*}} vector_sizes [16, 1]
// align2
// CHECK: transform.structured.vectorize %{{.*}} vector_sizes [8, 2]
// align2
// CHECK: transform.structured.vectorize %{{.*}} vector_sizes [64, 2]

// WITH_OPTIONS_2-LABEL: func @matmul_2

// WITH_OPTIONS_3-LABEL: func @matmul_2

// -----

hal.executable @matmul_3 {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @matmul_3 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_3() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2556xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2556xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2556xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2556xf32>> -> tensor<2048x2556xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2556xf32>> -> tensor<2556x2556xf32>
      %5 = tensor.empty() : tensor<2048x2556xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x2556xf32>) -> tensor<2048x2556xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2048x2556xf32>, tensor<2556x2556xf32>) outs(%6 : tensor<2048x2556xf32>) -> tensor<2048x2556xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2556], strides = [1, 1] : tensor<2048x2556xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2556xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @matmul_3

// CHECK: transform.sequence  failures(propagate) {

// WITH_OPTIONS_2-LABEL: func @matmul_3

// WITH_OPTIONS_3-LABEL: func @matmul_3

// -----
hal.executable @matmul_4_partially_unaligned {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @matmul_4_partially_unaligned ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_4_partially_unaligned() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2044xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2044x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2044xf32>> -> tensor<2048x2044xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2044x1024xf32>> -> tensor<2044x1024xf32>
      %5 = tensor.empty() : tensor<2048x1024xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2048x2044xf32>, tensor<2044x1024xf32>) outs(%6 : tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1] : tensor<2048x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x1024xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @matmul_4_partially_unaligned

// CHECK: transform.structured.tile_using_for %tiled_op[0, 0, 16]

// Make sure we do not canonicalize because the result is still aligned.
// CHECK-NEXT: transform.structured.pad %tiled_linalg_op
// CHECK-SAME:   copy_back_op = "none"
// CHECK-SAME:   pack_paddings = [1, 1, 1]
// CHECK-SAME:   padding_dimensions = [0, 1, 2]
// CHECK-SAME:   padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]
// CHECK:      apply_patterns to %{{.*}} {
// CHECK:        transform.apply_patterns.canonicalization
// CHECK       }
// CHECK:      transform.iree.apply_licm
// CHECK:      transform.iree.apply_cse
// CHECK:      %[[RES_PAD:.+]] = get_producer_of_operand %{{.*}}[2]
// CHECK:      %[[RES_COPY:.+]] = transform.structured.rewrite_in_destination_passing_style %[[RES_PAD]]
// CHECK:      %[[LHS_PAD:.+]] = get_producer_of_operand %{{.*}}[0]
// CHECK:      %[[RHS_PAD:.+]] = get_producer_of_operand %{{.*}}[1]
// CHECK:      %[[TILED_LHS:.+]], %{{.*}} = transform.structured.tile_using_forall %[[LHS_PAD]]   num_threads [32, 4](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:      transform.structured.match ops{["scf.if"]}
// CHECK:      transform.scf.take_assumed_branch %{{.*}} take_else_branch
// CHECK:      %[[TILED_RHS:.+]], %{{.*}} = transform.structured.tile_using_forall %[[RHS_PAD]]   num_threads [4, 32](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:      transform.structured.match ops{["scf.if"]}
// CHECK:      transform.scf.take_assumed_branch %{{.*}} take_else_branch
// CHECK:      transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK:      transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK:        transform.apply_patterns.canonicalization
// CHECK       }
// CHECK:      transform.iree.apply_licm
// CHECK:      transform.iree.apply_cse

// alignLhs
// CHECK:      transform.structured.vectorize %[[TILED_LHS]] vector_sizes [4, 4]
// alignRhs
// CHECK:      transform.structured.vectorize %[[TILED_RHS]] vector_sizes [4, 4]

// CHECK:      transform.apply_patterns.vector.lower_masks
// CHECK:      transform.apply_patterns.vector.materialize_masks

// WITH_OPTIONS_2-LABEL: func @matmul_4_partially_unaligned

// WITH_OPTIONS_3-LABEL: func @matmul_4_partially_unaligned

// -----
hal.executable @aligned_matmul {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @aligned_matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @aligned_matmul() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xf32>> -> tensor<2048x2048xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xf32>> -> tensor<2048x2048xf32>
      %5 = tensor.empty() : tensor<2048x2048xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%6 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @aligned_matmul

// Block level is the same for aligned.
// CHECK: transform.structured.tile_using_for %tiled_op[0, 0, 16]

// Make sure we do not canonicalize if the result is aligned to avoid folding the extract_slice on the iterator.
// CHECK-NEXT: transform.structured.pad %tiled_linalg_op
// CHECK-SAME:   copy_back_op = "none"
// CHECK-SAME:   pack_paddings = [1, 1, 1]
// CHECK-SAME:   padding_dimensions = [0, 1, 2]
// CHECK-SAME:   padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]

// Canonicalization is currently required here to enable pad to dps to produce linalg.copy ops.
// CHECK:      apply_patterns to %{{.*}} {
// CHECK:        transform.apply_patterns.canonicalization
// CHECK       }
// CHECK:      transform.iree.apply_licm
// CHECK:      transform.iree.apply_cse
// CHECK:      %[[RES_PAD:.+]] = get_producer_of_operand %{{.*}}[2]
// CHECK:      %[[RES_COPY:.+]] = transform.structured.rewrite_in_destination_passing_style %[[RES_PAD]]
// CHECK:      %[[LHS_PAD:.+]] = get_producer_of_operand %{{.*}}[0]
// CHECK:      %[[RHS_PAD:.+]] = get_producer_of_operand %{{.*}}[1]
// CHECK:      %[[LHS_COPY:.+]] = transform.structured.rewrite_in_destination_passing_style %[[LHS_PAD]]
// CHECK:      %[[RHS_COPY:.+]] = transform.structured.rewrite_in_destination_passing_style %[[RHS_PAD]]
// CHECK:      transform.structured.tile_using_forall %[[LHS_COPY]]   num_threads [32, 4](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:      transform.structured.tile_using_forall %[[RHS_COPY]]   num_threads [4, 32](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK:      transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK:      transform.structured.tile_using_forall %{{.*}}   num_threads [2, 2](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK:        transform.apply_patterns.canonicalization
// CHECK       }
// CHECK:      transform.iree.apply_licm
// CHECK:      transform.iree.apply_cse

// Verify we don't go down the path without the flag.
// WITH_OPTIONS-LABEL: func @aligned_matmul

// WITH_OPTIONS-NOT: transform.sequence  failures(propagate) {

// WITH_OPTIONS_2-LABEL: func @aligned_matmul

// WITH_OPTIONS_3-LABEL: func @aligned_matmul

// -----

hal.executable @matmul_5_small {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @matmul_5_small ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_5_small() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x2044xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2044x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x1024xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 2044], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2044xf32>> -> tensor<2x2044xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2044, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2044x1024xf32>> -> tensor<2044x1024xf32>
      %5 = tensor.empty() : tensor<2x1024xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x1024xf32>) -> tensor<2x1024xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2x2044xf32>, tensor<2044x1024xf32>) outs(%6 : tensor<2x1024xf32>) -> tensor<2x1024xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2, 1024], strides = [1, 1] : tensor<2x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x1024xf32>>
      return
    }
  }
}
}

// CHECK:       iree_codegen.translation_info<LLVMGPUMatmulSimt>
// CHECK-LABEL: func @matmul_5_small

// This matmul is considered "too small"/"degenerate" for a tensor core strategy,
// just fallback to the simt strategy.

// WITH_OPTIONS_2-LABEL: func @matmul_5_small

// WITH_OPTIONS_3-LABEL: func @matmul_5_small

// SMALL-LABEL: func @matmul_5_small
// SMALL: transform.sequence
// SMALL-NOT: mma
// SMALL-NOT: wmma

// -----

hal.executable @f16_matmul {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @f16_matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @f16_matmul() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2052x2556xf16>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2052xf16>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2052x2052xf16>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2052, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2052x2556xf16>> -> tensor<2052x2556xf16>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2052], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2052xf16>> -> tensor<2556x2052xf16>
      %5 = tensor.empty() : tensor<2052x2052xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2052x2052xf16>) -> tensor<2052x2052xf16>
      %7 = linalg.matmul ins(%3, %4 : tensor<2052x2556xf16>, tensor<2556x2052xf16>) outs(%6 : tensor<2052x2052xf16>) -> tensor<2052x2052xf16>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2052, 2052], strides = [1, 1] : tensor<2052x2052xf16> -> !flow.dispatch.tensor<writeonly:tensor<2052x2052xf16>>
      return
    }
  }
}
}

// CHECK:       iree_codegen.translation_info<LLVMGPUMatmulSimt>
// CHECK-LABEL: func @f16_matmul
// CHECK-NOT: transform.sequence

// WITH_OPTIONS_2-LABEL: func @f16_matmul

// WITH_OPTIONS_3-LABEL: func @f16_matmul


// -----

hal.executable @int8_matmul {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @int8_matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @int8_matmul() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0 : i8
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x2556xi8>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2052xi8>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x2052xi8>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x2556xi8>> -> tensor<4x2556xi8>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2052], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2052xi8>> -> tensor<2556x2052xi8>
      %5 = tensor.empty() : tensor<4x2052xi8>
      %6 = linalg.fill ins(%cst : i8) outs(%5 : tensor<4x2052xi8>) -> tensor<4x2052xi8>
      %7 = linalg.matmul ins(%3, %4 : tensor<4x2556xi8>, tensor<2556x2052xi8>) outs(%6 : tensor<4x2052xi8>) -> tensor<4x2052xi8>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [4, 2052], strides = [1, 1] : tensor<4x2052xi8> -> !flow.dispatch.tensor<writeonly:tensor<4x2052xi8>>
      return
    }
  }
}
}

// SMALL-LABEL: func @int8_matmul
// SMALL: transform.sequence
// SMALL-NOT: mma
// SMALL-NOT: wmma

// CHECK-LABEL: func @int8_matmul
// CHECK-NOT: transform.sequence

// WITH_OPTIONS-LABEL: func @int8_matmul
// WITH_OPTIONS-NOT: transform.sequence

// WITH_OPTIONS_2-LABEL: func @int8_matmul
// WITH_OPTIONS_2-NOT: transform.sequence

// WITH_OPTIONS_3-LABEL: func @int8_matmul
// WITH_OPTIONS_3-NOT: transform.sequence
