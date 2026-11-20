// RUN: iree-opt --iree-gpu-test-target=sm_80 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-configuration-preprocessing-pipeline, builtin.module(iree-codegen-llvmgpu-configuration-pipeline, iree-codegen-llvmgpu-nvvm-lowering-pipeline), iree-codegen-translation-postprocessing-pipeline)))" %s | FileCheck %s

// Verify that elementwise FP8-to-FP32 conversion does not leave FP8 values or
// unrealized casts in the final CUDA LLVM module. This is a general CUDA Core
// lowering and must not depend on FP8 Tensor Core support.

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @fp8_to_f32 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @fp8_to_f32 ordinal(0) layout(#pipeline_layout)
        count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @convert_e4m3fn() {
        %c0 = arith.constant 0 : index
        %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf8E4M3FN>>
        %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
        %input_tensor = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0], sizes = [1024], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf8E4M3FN>> -> tensor<1024xf8E4M3FN>
        %empty = tensor.empty() : tensor<1024xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%input_tensor : tensor<1024xf8E4M3FN>) outs(%empty : tensor<1024xf32>) {
        ^bb0(%value: f8E4M3FN, %unused: f32):
          %extended = arith.extf %value : f8E4M3FN to f32
          linalg.yield %extended : f32
        } -> tensor<1024xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [1024], strides = [1]
            : tensor<1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
        return
      }

      func.func @convert_e5m2() {
        %c0 = arith.constant 0 : index
        %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf8E5M2>>
        %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
        %input_tensor = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0], sizes = [1024], strides = [1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf8E5M2>> -> tensor<1024xf8E5M2>
        %empty = tensor.empty() : tensor<1024xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%input_tensor : tensor<1024xf8E5M2>) outs(%empty : tensor<1024xf32>) {
        ^bb0(%value: f8E5M2, %unused: f32):
          %extended = arith.extf %value : f8E5M2 to f32
          linalg.yield %extended : f32
        } -> tensor<1024xf32>
        iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [1024], strides = [1]
            : tensor<1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @fp8_to_f32
// CHECK: builtin.module
// CHECK-LABEL: llvm.func @convert_e4m3fn
// CHECK: llvm.zext
// CHECK: llvm.bitcast
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: f8E4M3FN
// CHECK-NOT: f8E5M2
// CHECK-LABEL: llvm.func @convert_e5m2
// CHECK: llvm.zext
// CHECK: llvm.bitcast
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: f8E4M3FN
// CHECK-NOT: f8E5M2
