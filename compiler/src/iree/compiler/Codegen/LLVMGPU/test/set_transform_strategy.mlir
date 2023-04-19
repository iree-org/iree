// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" --iree-codegen-llvmgpu-enable-transform-dialect-jit --iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy | FileCheck %s

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

// One of this matmul's dimensions is divisible by 64/64/16, we currently bail on such cases.
// CHECK-NOT: transform.sequence

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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2049x2556xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2556x2556xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2049x2556xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2049, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2049x2556xf32>> -> tensor<2049x2556xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2556, 2556], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2556x2556xf32>> -> tensor<2556x2556xf32>
      %5 = tensor.empty() : tensor<2049x2556xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2049x2556xf32>) -> tensor<2049x2556xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2049x2556xf32>, tensor<2556x2556xf32>) outs(%6 : tensor<2049x2556xf32>) -> tensor<2049x2556xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2049, 2556], strides = [1, 1] : tensor<2049x2556xf32> -> !flow.dispatch.tensor<writeonly:tensor<2049x2556xf32>>
      return
    }
  }
}
}

// One of this matmul's dimensions is not divisible by 4, we currently bail on such cases.
// CHECK-NOT: transform.sequence

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

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.iree.match_callback failures(propagate) "matmul"
// CHECK: transform.iree.tile_to_forall_and_workgroup_count_region %{{.*}} num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.structured.tile %{{.*}}[0, 0, 16]
// CHECK: transform.structured.pad %{{.*}} {pack_paddings = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK: transform.structured.hoist_pad %{{.}} by 1 loops
// CHECK: transform.structured.insert_slice_to_copy %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [32, 4] tile_sizes [](mapping = [#gpu.thread<x>, #gpu.thread<y>])
// CHECK:   transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!pdl.operation) -> ()
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [4, 32] tile_sizes [](mapping = [#gpu.thread<y>, #gpu.thread<x>])
// CHECK: transform.scf.take_assumed_branch %{{.*}} take_else_branch : (!pdl.operation) -> ()
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [4, 32] tile_sizes [](mapping = [#gpu.thread<y>, #gpu.thread<x>])
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 2] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_to_forall_op %{{.*}}   num_threads [2, 2] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [4, 4]
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [4, 4]
// CHECK: transform.structured.masked_vectorize %{{.*}} vector_sizes [32, 4]
// CHECK: transform.vector.lower_masked_transfers %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.structured.vectorize %{{.*}}
// CHECK: transform.iree.eliminate_empty_tensors %{{.*}}
// CHECK: transform.iree.bufferize {target_gpu} %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.erase_hal_descriptor_type_from_memref %{{.*}} : (!pdl.operation) -> ()
// CHECK: transform.iree.forall_to_workgroup %{{.*}} : (!pdl.operation) -> ()
// CHECK: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [32, 4, 1] warp_dims = [2, 2, 1] : (!pdl.operation) -> ()
// CHECK: transform.iree.hoist_static_alloc %{{.*}} : (!pdl.operation) -> ()
// CHECK: transform.iree.apply_patterns %{{.*}} {fold_memref_aliases} : (!pdl.operation) -> ()
// CHECK: transform.iree.apply_patterns %{{.*}} {extract_address_computations} : (!pdl.operation) -> ()
// CHECK: transform.iree.apply_patterns %{{.*}} {unroll_vectors_gpu_wmma} : (!pdl.operation) -> ()
// CHECK: transform.structured.hoist_redundant_vector_transfers %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.apply_buffer_optimizations %{{.*}} : (!pdl.operation) -> ()
// CHECK: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_wmma} : (!pdl.operation) -> ()
// CHECK: transform.iree.apply_patterns %{{.*}} {fold_memref_aliases} : (!pdl.operation) -> ()
// CHECK: transform.memref.multibuffer %{{.*}} {factor = 3 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !pdl.operation
// CHECK: transform.vector.transfer_to_scf %{{.*}}   max_transfer_rank = 1 full_unroll = true : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.create_async_groups %{{.*}} {use_mma_sync = false} : (!pdl.operation) -> ()
// CHECK: transform.iree.pipeline_shared_memory_copies %{{.*}} {depth = 3 : i64} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.vector.lower_masks %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.vector.materialize_masks %{{.*}} : (!pdl.operation) -> !pdl.operation
// CHECK: transform.iree.apply_patterns %{{.*}} {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization} : (!pdl.operation) -> ()
