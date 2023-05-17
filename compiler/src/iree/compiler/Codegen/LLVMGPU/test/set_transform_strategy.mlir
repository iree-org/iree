// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" --iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy | FileCheck %s
// Check that setting the command line options affect the transform
// strategy as expected.
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" --iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy \
// RUN: -td-matmul-strategy-blk-size-x=256 \
// RUN: -td-matmul-strategy-blk-size-y=64 \
// RUN: -td-matmul-strategy-blk-size-z=1 \
// RUN: -td-matmul-strategy-reduc-size=8 \
// RUN: -td-matmul-strategy-num-threads-x=32 \
// RUN: -td-matmul-strategy-num-threads-y=4 \
// RUN: -td-matmul-strategy-num-threads-z=1 \
// RUN: -td-matmul-strategy-num-warps-x=1 \
// RUN: -td-matmul-strategy-num-warps-y=4 \
// RUN: -td-matmul-strategy-num-warps-z=1 \
// RUN: -td-matmul-strategy-use-async-copies=true \
// RUN: -td-matmul-strategy-use-mma-sync=true \
// RUN: -td-matmul-strategy-pipeline-depth=5 \
// RUN: | FileCheck --check-prefix=WITH_OPTIONS %s

hal.executable @matmul {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul() {
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

// CHECK-LABEL: func @matmul

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.iree.match_callback failures(propagate) "matmul"
// CHECK: transform.structured.tile_to_forall_op %{{.*}} num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.iree.populate_workgroup_count_region_using_num_threads_slice
// CHECK: transform.structured.tile %{{.*}}[0, 0, 16]
// CHECK: transform.structured.pad %{{.*}} {pack_paddings = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK: transform.structured.hoist_pad %{{.}} by 1 loops
// CHECK: transform.structured.insert_slice_to_copy %{{.*}} : (!transform.any_op) -> !transform.any_op
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [32, 4] tile_sizes [](mapping = [#gpu.linear<x>, #gpu.linear<y>])
// CHECK:   transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [4, 32] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
// CHECK: transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [4, 32] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 2] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 2] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [4, 4]
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [4, 4]
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [32, 4]
// CHECK: transform.vector.lower_masked_transfers %{{.*}}
// CHECK: transform.structured.vectorize %{{.*}}
// CHECK: transform.iree.eliminate_empty_tensors %{{.*}}
// CHECK: transform.iree.bufferize {target_gpu} %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.erase_hal_descriptor_type_from_memref %{{.*}}
// CHECK: transform.iree.forall_to_workgroup %{{.*}}
// CHECK: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [64, 2, 1] warp_dims = [2, 2, 1]
// CHECK: transform.iree.hoist_static_alloc %{{.*}}
// CHECK: transform.iree.apply_patterns %{{.*}} {fold_memref_aliases}
// CHECK: transform.iree.apply_patterns %{{.*}} {extract_address_computations}
// CHECK: transform.iree.apply_patterns %{{.*}} {unroll_vectors_gpu_wmma}
// CHECK: transform.structured.hoist_redundant_vector_transfers %{{.*}}
// CHECK: transform.iree.apply_buffer_optimizations %{{.*}} : (!pdl.operation) -> ()
// CHECK: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_wmma} : (!pdl.operation) -> ()
// CHECK: transform.iree.apply_patterns %{{.*}} {fold_memref_aliases}
// CHECK: transform.memref.multibuffer %{{.*}} {factor = 3 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !pdl.operation
// CHECK: transform.vector.transfer_to_scf %{{.*}}   max_transfer_rank = 1 full_unroll = true : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.create_async_groups %{{.*}} {use_mma_sync = false} : (!pdl.operation) -> ()
// CHECK: transform.iree.pipeline_shared_memory_copies %{{.*}} {depth = 3 : i64} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.vector.lower_masks %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.vector.materialize_masks %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.apply_patterns %{{.*}} {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization}


// WITH_OPTIONS-LABEL: func @matmul

// WITH_OPTIONS: transform.sequence  failures(propagate) {
// WITH_OPTIONS: transform.iree.match_callback failures(propagate) "matmul"
// Tile sizes are set by td-matmul-strategy-blk-size-XX.
// WITH_OPTIONS: transform.structured.tile_to_forall_op %{{.*}} num_threads [] tile_sizes [256, 64](mapping = [#gpu.block<y>, #gpu.block<x>])
// WITH_OPTIONS: transform.structured.fuse_into_containing_op
// WITH_OPTIONS: transform.iree.populate_workgroup_count_region_using_num_threads_slice
// The tiling is affected by td-matmul-strategy-reduc-size: 8.
// WITH_OPTIONS: transform.structured.tile %{{.*}}[0, 0, 8]
// WITH_OPTIONS: transform.structured.pad %{{.*}} {pack_paddings = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// WITH_OPTIONS: transform.structured.hoist_pad %{{.}} by 1 loops
// WITH_OPTIONS: transform.structured.insert_slice_to_copy %{{.*}} : (!transform.any_op) -> !transform.any_op
// WITH_OPTIONS: transform.structured.tile_to_forall_op %{{.*}}   num_threads [64, 2] tile_sizes [](mapping = [#gpu.linear<x>, #gpu.linear<y>])
// WITH_OPTIONS:   transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// WITH_OPTIONS: transform.structured.tile_to_forall_op %{{.*}}   num_threads [8, 16] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
// WITH_OPTIONS: transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!transform.any_op) -> ()
// WITH_OPTIONS: transform.structured.tile_to_forall_op %{{.*}}   num_threads [8, 16] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
// WITH_OPTIONS: transform.structured.tile_to_forall_op %{{.*}}   num_threads [1, 4] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// WITH_OPTIONS: transform.structured.tile_to_forall_op %{{.*}}   num_threads [1, 4] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// WITH_OPTIONS: transform.structured.masked_vectorize %{{.*}} vector_sizes [4, 4]
// WITH_OPTIONS: transform.structured.masked_vectorize %{{.*}} vector_sizes [1, 4]
// WITH_OPTIONS: transform.structured.masked_vectorize %{{.*}} vector_sizes [32, 4]
// WITH_OPTIONS: transform.vector.lower_masked_transfers %{{.*}}
// WITH_OPTIONS: transform.structured.vectorize %{{.*}}
// WITH_OPTIONS: transform.iree.eliminate_empty_tensors %{{.*}}
// WITH_OPTIONS: transform.iree.bufferize {target_gpu} %{{.*}} : (!pdl.operation) -> !pdl.operation
// WITH_OPTIONS: transform.iree.erase_hal_descriptor_type_from_memref %{{.*}}
// WITH_OPTIONS: transform.iree.forall_to_workgroup %{{.*}}
// The workgroup dimensions are controled by td-matmul-strategy-num-threads-XX.
// The warp dimensions are controled by td-matmul-strategy-num-warps-XX.
// WITH_OPTIONS: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [32, 4, 1] warp_dims = [1, 4, 1]
// WITH_OPTIONS: transform.iree.hoist_static_alloc %{{.*}}
// WITH_OPTIONS: transform.iree.apply_patterns %{{.*}} {fold_memref_aliases}
// WITH_OPTIONS: transform.iree.apply_patterns %{{.*}} {extract_address_computations}
// The unroll attribute should match td-matmul-use-mma-sync, for true: mma_sync,
// for false:_wmma.
// WITH_OPTIONS: transform.iree.apply_patterns %{{.*}} {unroll_vectors_gpu_mma_sync}
// WITH_OPTIONS: transform.structured.hoist_redundant_vector_transfers %{{.*}}
// WITH_OPTIONS: transform.iree.apply_buffer_optimizations %{{.*}} : (!pdl.operation) -> ()
// The attribute should match td-matmul-use-mma-sync.
// WITH_OPTIONS: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_mma_sync} : (!pdl.operation) -> ()
// WITH_OPTIONS: transform.iree.apply_patterns %{{.*}} {fold_memref_aliases}
// The multibuffer pass is only run when we set use-async-copies.
// The factor should match td-matmul-strategy-pipeline-depth: 5.
// WITH_OPTIONS: transform.memref.multibuffer %{{.*}} {factor = 5 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !pdl.operation
// WITH_OPTIONS: transform.vector.transfer_to_scf %{{.*}}   max_transfer_rank = 1 full_unroll = true : (!pdl.operation) -> !pdl.operation
// The attribute should match td-matmul-use-mma-sync.
// WITH_OPTIONS: transform.iree.create_async_groups %{{.*}} {use_mma_sync = true} : (!pdl.operation) -> ()
// The depth should match td-matmul-strategy-pipeline-depth: 5.
// WITH_OPTIONS: transform.iree.pipeline_shared_memory_copies %{{.*}} {depth = 5 : i64} : (!pdl.operation) -> !pdl.operation
// WITH_OPTIONS: transform.vector.lower_masks %{{.*}} : (!pdl.operation) -> !pdl.operation
// WITH_OPTIONS: transform.vector.materialize_masks %{{.*}} : (!pdl.operation) -> !pdl.operation
// WITH_OPTIONS: transform.iree.apply_patterns %{{.*}} {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization}

// -----

hal.executable @matmul {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul() {
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

// CHECK-LABEL: func @matmul

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.iree.match_callback failures(propagate) "matmul"
// CHECK: transform.structured.tile_to_forall_op %{{.*}} num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
// CHECK: transform.iree.populate_workgroup_count_region_using_num_threads_slice
// CHECK: transform.structured.tile %{{.*}}[0, 0, 16]
// align1
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [8, 16] tile_sizes [](mapping = [#gpu.linear<x>, #gpu.linear<y>])
// align2
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 64] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
// align2
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 64] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 2] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 2] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// align1
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [16, 1]
// align2
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [8, 2]
// align2
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [64, 2]

// -----
hal.executable @matmul {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul() {
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

// CHECK-LABEL: func @matmul

// CHECK: transform.sequence  failures(propagate) {

// -----
hal.executable @matmul {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul() {
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

// CHECK-LABEL: func @matmul

// "Enough" of this matmul's dimensions are divisible by 64/64/16.
// We currently bail on such cases because at least one of the paddings involved
// in the strategy fold away and result in the strategy failing to apply.
// In the future we should also support this case but for now we are missing the 
// generalization along this axis.
// CHECK-NOT: transform.sequence

// -----
hal.executable @matmul_partially_unaligned {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul_partially_unaligned ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_partially_unaligned() {
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

// CHECK:       iree_codegen.translation_info<LLVMGPUMatmulSimt>
// CHECK-LABEL: func @matmul_partially_unaligned

// "Enough" of this matmul's dimensions are divisible by 64/64/16.
// We currently bail on such cases because at least one of the paddings involved
// in the strategy fold away and result in the strategy failing to apply.
// In the future we should also support this case but for now we are missing the
// generalization along this axis.
// CHECK-NOT: transform.sequence

// -----

hal.executable @matmul_too_small {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @matmul_too_small ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @matmul_too_small() {
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
// CHECK-LABEL: func @matmul_too_small

// This matmul is considered "too small"/"degenerate" for a tensor core strategy,
// just fallback to the simt strategy.
// CHECK-NOT: transform.sequence

// -----

hal.executable @f16_matmul {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
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
