// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" --iree-codegen-llvmgpu-enable-transform-dialect-implicit-gemm-strategy | FileCheck %s

hal.executable @nchw_convolution {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @nchw_convolution ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @nchw_convolution() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x128x258x258xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x128x3x3xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 128, 258, 258], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x128x258x258xf32>> -> tensor<8x128x258x258xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [256, 128, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128x3x3xf32>> -> tensor<256x128x3x3xf32>
      %5 = tensor.empty() : tensor<8x256x256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                ins(%3, %4 : tensor<8x128x258x258xf32>, tensor<256x128x3x3xf32>) outs(%6 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 256, 256, 256], strides = [1, 1, 1, 1] : tensor<8x256x256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @nchw_convolution

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.iree.match_callback failures(propagate) "convolution"
// CHECK: transform.structured.convert_conv2d_to_img2col
// CHECK: get_producer_of_operand %{{.*}}[0]
// CHECK: transform.apply_patterns.iree.bubble_collapse
// CHECK: transform.structured.tile_using_forall %{{.*}}   tile_sizes [1, 128, 128](mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>])
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.iree.populate_workgroup_count_region_using_num_threads_slice %{{.*}}
// CHECK: transform.structured.match ops{["linalg.fill"]}
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.structured.tile_using_for %{{.*}}[0, 0, 0, 16]
// CHECK: transform.structured.fuse_into_containing_op
// CHECK: transform.structured.pad %{{.*}} {copy_back_op = "none", pack_paddings = [1, 0, 1], pad_to_multiple_of = [1, 1, 1, 1], padding_dimensions = [0, 1, 2, 3], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK: transform.structured.match ops{["linalg.fill"]}
// CHECK: %[[RES:.+]] = get_producer_of_operand %{{.*}}[2]
// CHECK: transform.structured.rewrite_in_destination_passing_style %[[RES]]
// CHECK: %[[LHS:.+]] = get_producer_of_operand %{{.*}}[0]
// CHECK: %[[RHS:.+]] = get_producer_of_operand %{{.*}}[1]
// CHECK: transform.structured.rewrite_in_destination_passing_style %[[LHS]]
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [32, 4](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.structured.tile_using_forall %[[RHS]]   num_threads [1, 4, 32](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [1, 2, 2](mapping = [#gpu.warp<z>, #gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [1, 2, 2](mapping = [#gpu.warp<z>, #gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
// CHECK: transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
// CHECK: transform.apply_patterns.vector.cast_away_vector_leading_one_dim
// CHECK: transform.structured.vectorize_children_and_apply_patterns %{{.*}} {vectorize_nd_extract}
// CHECK: transform.iree.eliminate_empty_tensors
// CHECK: transform.iree.bufferize {target_gpu}
// CHECK: transform.memref.erase_dead_alloc_and_stores
// CHECK: transform.iree.forall_to_workgroup
// CHECK: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [64, 2, 1]
// CHECK: transform.iree.hoist_static_alloc %{{.*}}
// CHECK: transform.apply_patterns.memref.fold_memref_alias_ops
// CHECK: transform.apply_patterns.memref.extract_address_computations
// CHECK: transform.apply_patterns.iree.unroll_vectors_gpu_mma_sync
// CHECK: transform.structured.hoist_redundant_vector_transfers
// CHECK: transform.iree.vector.vector_to_mma_conversion %{{.*}} {use_mma_sync}

// -----

hal.executable @nhwc_convolution {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @nhwc_convolution ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @nhwc_convolution() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x258x258x128xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x128x256xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 258, 258, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x258x258x128xf32>> -> tensor<8x258x258x128xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x128x256xf32>> -> tensor<3x3x128x256xf32>
      %5 = tensor.empty() : tensor<8x256x256x256xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                ins(%3, %4 : tensor<8x258x258x128xf32>, tensor<3x3x128x256xf32>) outs(%6 : tensor<8x256x256x256xf32>) -> tensor<8x256x256x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 256, 256, 256], strides = [1, 1, 1, 1] : tensor<8x256x256x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x256x256x256xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @nhwc_convolution

// CHECK: transform.sequence  failures(propagate) {
// CHECK: transform.structured.tile_using_forall %{{.*}}   tile_sizes [1, 128, 128](mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>])
// CHECK: transform.structured.pad %{{.*}} {copy_back_op = "none", pack_paddings = [0, 1, 1], pad_to_multiple_of = [1, 1, 1, 1], padding_dimensions = [0, 1, 2, 3], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK: %[[RES:.+]] = get_producer_of_operand %{{.*}}[2]
// CHECK: transform.structured.rewrite_in_destination_passing_style %[[RES]]
// CHECK: %[[LHS:.+]] = get_producer_of_operand %{{.*}}[0]
// CHECK: %[[RHS:.+]] = get_producer_of_operand %{{.*}}[1]
// CHECK: transform.structured.rewrite_in_destination_passing_style %[[RHS]]
// CHECK: transform.structured.tile_using_forall %[[LHS]]   num_threads [1, 32, 4](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [4, 32](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [1, 2, 2](mapping = [#gpu.warp<z>, #gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.structured.tile_using_forall %{{.*}}   num_threads [1, 2, 2](mapping = [#gpu.warp<z>, #gpu.warp<y>, #gpu.warp<x>])
// CHECK: transform.iree.map_nested_forall_to_gpu_threads %{{.*}} workgroup_dims = [64, 2, 1]


// -----

hal.executable @unaligned_convolution {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>) {
  hal.executable.export public @unaligned_convolution ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @unaligned_convolution() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x258x258x132xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x132x264xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x256x256x264xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 258, 258, 132], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x258x258x132xf32>> -> tensor<8x258x258x132xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 132, 264], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x132x264xf32>> -> tensor<3x3x132x264xf32>
      %5 = tensor.empty() : tensor<8x256x256x264xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x256x256x264xf32>) -> tensor<8x256x256x264xf32>
      %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
                ins(%3, %4 : tensor<8x258x258x132xf32>, tensor<3x3x132x264xf32>) outs(%6 : tensor<8x256x256x264xf32>) -> tensor<8x256x256x264xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 256, 256, 264], strides = [1, 1, 1, 1] : tensor<8x256x256x264xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x256x256x264xf32>>
      return
    }
  }
}
}

// CHECK:       #iree_codegen.translation_info<LLVMGPUVectorize>
// CHECK-LABEL: func @unaligned_convolution

// Currently padding on the img2col op is not supported so bail out for unaligned.
// CHECK-NOT: transform.sequence
