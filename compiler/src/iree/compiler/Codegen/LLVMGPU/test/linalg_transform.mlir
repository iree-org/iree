// RUN: iree-opt %s  --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' \
// RUN: --iree-codegen-llvmgpu-use-transform-dialect=%p/transform_dialect_codegen_bufferize_spec.mlir | \
// RUN: FileCheck %s

// RUN: iree-opt %s  --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' \
// RUN: --iree-codegen-llvmgpu-use-transform-dialect=%p/transform_dialect_codegen_foreach_to_gpu_spec.mlir | \
// RUN: FileCheck %s --check-prefix=FOREACH-TO-GPU

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>]}>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable private @matmul_static_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {

      // FOREACH-TO-GPU: hal.executable.export {{.*}} workgroup_size = [7 : index, 9 : index, 1 : index]
      hal.executable.export public @matmul_static_dispatch_0 ordinal(0) layout(#executable_layout)
      builtin.module {
        func.func @matmul_static_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:250x500xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:500x1020xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:250x1020xf32>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x500xf32> -> tensor<250x500xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !flow.dispatch.tensor<readonly:500x1020xf32> -> tensor<500x1020xf32>

          %50 = linalg.init_tensor [250, 1020] : tensor<250x1020xf32>
          %cst = arith.constant 0.000000e+00 : f32
          %5 = linalg.fill ins(%cst : f32) outs(%50 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

          //      CHECK: memref.assume_alignment %{{.*}}, 64 : memref<250x1020xf32>
          // CHECK-NEXT: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32>)
          // CHECK-NEXT: linalg.matmul{{.*}}ins(%{{.*}} : memref<250x500xf32>, memref<500x1020xf32>) outs(%{{.*}} : memref<250x1020xf32>)
          // CHECK-NEXT: return


          // FOREACH-TO-GPU: %[[TIDX:.*]] = gpu.thread_id  x
          // FOREACH-TO-GPU: %[[TIDY:.*]] = gpu.thread_id  y
          // FOREACH-TO-GPU: affine.min #{{.*}}()[%[[TIDX]]]
          // FOREACH-TO-GPU: affine.min #{{.*}}()[%[[TIDY]]]
          // FOREACH-TO-GPU: affine.apply #{{.*}}()[%[[TIDX]]]
          // FOREACH-TO-GPU: %[[svA:.*]] = memref.subview {{.*}} : memref<250x500xf32> to memref<?x500xf32
          // FOREACH-TO-GPU: affine.apply #{{.*}}()[%[[TIDY]]]
          // FOREACH-TO-GPU: %[[svB:.*]] = memref.subview {{.*}} : memref<500x1020xf32> to memref<500x?xf32
          // FOREACH-TO-GPU: %[[svC:.*]] = memref.subview {{.*}} : memref<250x1020xf32> to memref<?x?xf32
          // FOREACH-TO-GPU: linalg.matmul ins(%[[svA]], %[[svB]] : memref<?x500xf32{{.*}}>, memref<500x?xf32{{.*}}>) outs(%[[svC]] : memref<?x?xf32{{.*}}>)

          %6 = linalg.matmul ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : tensor<250x1020xf32> -> !flow.dispatch.tensor<readwrite:250x1020xf32>
          return
        }
      }
    }
  }
}
