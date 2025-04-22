// RUN: iree-opt %s  --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline, func.func(iree-llvmgpu-lower-executable-target))" \
// RUN:     --iree-gpu-test-target=sm_60 \
// RUN:     --iree-codegen-transform-dialect-library=%p/transform_dialect_codegen_bufferize_spec.mlir@__transform_main | \
// RUN: FileCheck %s

// RUN: iree-opt %s  --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline, func.func(iree-llvmgpu-lower-executable-target))" \
// RUN:     --iree-gpu-test-target=sm_60 \
// RUN:     --iree-codegen-transform-dialect-library=%p/transform_dialect_codegen_foreach_to_gpu_spec.mlir@__transform_main | \
// RUN: FileCheck %s --check-prefix=FOREACH-TO-GPU

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
func.func @matmul_static_dispatch_0() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<250x500xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<500x1020xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<250x1020xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<500x1020xf32>> -> tensor<500x1020xf32>
  %5 = tensor.empty() : tensor<250x1020xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

  //      CHECK: memref.assume_alignment %{{.*}}, 64 : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>
  // CHECK-NEXT: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>)
  // CHECK-NEXT: linalg.matmul{{.*}}ins(%{{.*}} : memref<250x500xf32, #hal.descriptor_type<storage_buffer>>, memref<500x1020xf32, #hal.descriptor_type<storage_buffer>>) outs(%{{.*}} : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>)
  // CHECK-NEXT: return

  // workgroup_size is explicitly set to [10, 11].
  // FOREACH-TO-GPU: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = None workgroup_size = [10, 11, 1] subgroup_size = 32>
  // FOREACH-TO-GPU: func.func @matmul_static_dispatch_0()
  // FOREACH-TO-GPU-SAME: translation_info = #translation
  // FOREACH-TO-GPU-DAG: %[[C0:.*]] = arith.constant 0 : index
  // FOREACH-TO-GPU-DAG: %[[C1:.*]] = arith.constant 1 : index
  // FOREACH-TO-GPU-DAG: %[[C5:.*]] = arith.constant 5 : index
  // FOREACH-TO-GPU-DAG: %[[C7:.*]] = arith.constant 7 : index
  // FOREACH-TO-GPU-DAG: %[[C9:.*]] = arith.constant 9 : index
  // FOREACH-TO-GPU-DAG: %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
  // FOREACH-TO-GPU: %[[TIDX:.*]] = gpu.thread_id  x
  // FOREACH-TO-GPU: %[[TIDY:.*]] = gpu.thread_id  y
  //
  // Fill is tiled by 5x1 with thread_dim_mapping = [1, 0, 2], predicate appropriately.
  // FOREACH-TO-GPU: %[[LT1:.*]]  = arith.cmpi ult, %[[TIDX]], %[[C1]] : index
  // FOREACH-TO-GPU: %[[LT5:.*]]  = arith.cmpi ult, %[[TIDY]], %[[C5]] : index
  // FOREACH-TO-GPU: %[[COND:.*]]  = arith.andi %[[LT1]], %[[LT5]] : i1
  // FOREACH-TO-GPU: scf.if %[[COND]] {
  // FOREACH-TO-GPU:   affine.apply #{{.*}}()[%[[TIDY]]]
  // FOREACH-TO-GPU:   linalg.fill
  // FOREACH-TO-GPU: }
  // FOREACH-TO-GPU: gpu.barrier
  //
  // Matmul is tiled by 7x9 with identity (omitted) thread_dim_mapping, predicate appropriately.
  // FOREACH-TO-GPU: %[[LT7:.*]]  = arith.cmpi ult, %[[TIDX]], %[[C7]] : index
  // FOREACH-TO-GPU: %[[LT9:.*]]  = arith.cmpi ult, %[[TIDY]], %[[C9]] : index
  // FOREACH-TO-GPU: %[[COND2:.*]]  = arith.andi %[[LT7]], %[[LT9]] : i1
  // FOREACH-TO-GPU: scf.if %[[COND2]] {
  // FOREACH-TO-GPU:   affine.min #{{.*}}()[%[[TIDX]]]
  // FOREACH-TO-GPU:   affine.min #{{.*}}()[%[[TIDY]]]
  // FOREACH-TO-GPU-DAG:   affine.apply #{{.*}}()[%[[TIDX]]]
  // FOREACH-TO-GPU-DAG:   %[[svA:.*]] = memref.subview {{.*}} : memref<250x500xf32{{.*}}> to memref<?x500xf32
  // FOREACH-TO-GPU-DAG:   affine.apply #{{.*}}()[%[[TIDY]]]
  // FOREACH-TO-GPU-DAG:   %[[svB:.*]] = memref.subview {{.*}} : memref<500x1020xf32{{.*}}> to memref<500x?xf32
  // FOREACH-TO-GPU-DAG:   %[[svC:.*]] = memref.subview {{.*}} : memref<250x1020xf32{{.*}}> to memref<?x?xf32
  // FOREACH-TO-GPU:   linalg.matmul ins(%[[svA]], %[[svB]] : memref<?x500xf32{{.*}}>, memref<500x?xf32{{.*}}>) outs(%[[svC]] : memref<?x?xf32{{.*}}>)
  // FOREACH-TO-GPU: }
  // FOREACH-TO-GPU: gpu.barrier

  %7 = linalg.matmul ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%6 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : tensor<250x1020xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<250x1020xf32>>
  return
}
