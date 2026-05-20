// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), builtin.module(iree-codegen-llvmgpu-rocdl-lowering-pipeline{use-spirv=true}))))" \
// RUN:   %s | FileCheck %s

// Verify that a simple element-wise op lowers through the ROCDL pipeline with
// SPIR-V preparation (spir_kernel CC, address space remapping, correct triple).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @simpleMath_ex_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-spirv-fb">) {
    hal.executable.export public @add_dispatch_0 layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add_dispatch_0() {
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
        %3 = tensor.empty() : tensor<16xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4, %5 : tensor<16xf32>, tensor<16xf32>) outs(%3 : tensor<16xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<16xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @simpleMath_ex_dispatch_0
//       CHECK:   hal.executable.variant public @rocm
//       CHECK:   module attributes
//  CHECK-SAME:     llvm.target_triple = "spirv64-amd-amdhsa"
//       CHECK:   llvm.func spir_kernelcc @add_dispatch_0
//       CHECK:     llvm.fadd
//   CHECK-NOT:     amdgpu_kernel
//   CHECK-NOT:     addr_space = 7

// -----

// Verify matmul lowers through the SPIR-V pipeline without buffer instructions.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @matmul_dispatch_0 {
  hal.executable.variant @rocm target(<"rocm", "rocm-spirv-fb">) {
    hal.executable.export public @matmul_dispatch_0 layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_dispatch_0() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1024xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x512xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1024xf32>> -> tensor<2048x1024xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
        %5 = tensor.empty() : tensor<2048x512xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x512xf32>) -> tensor<2048x512xf32>
        %7 = linalg.matmul ins(%3, %4 : tensor<2048x1024xf32>, tensor<1024x512xf32>) outs(%6 : tensor<2048x512xf32>) -> tensor<2048x512xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : tensor<2048x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x512xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @matmul_dispatch_0
//       CHECK:   module attributes
//  CHECK-SAME:     llvm.target_triple = "spirv64-amd-amdhsa"
//       CHECK:   llvm.func spir_kernelcc @matmul_dispatch_0
//   CHECK-NOT:     addr_space = 7
