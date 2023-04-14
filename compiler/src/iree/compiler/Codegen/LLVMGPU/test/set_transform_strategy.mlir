// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" --iree-codegen-llvmgpu-enable-transform-dialect-jit %s | FileCheck %s

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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2560x2560xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2560x2560xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2560x2560xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2560, 2560], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2560x2560xf32>> -> tensor<2560x2560xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2560, 2560], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2560x2560xf32>> -> tensor<2560x2560xf32>
      %5 = tensor.empty() : tensor<2560x2560xf32>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2560x2560xf32>) -> tensor<2560x2560xf32>
      %7 = linalg.matmul ins(%3, %4 : tensor<2560x2560xf32>, tensor<2560x2560xf32>) outs(%6 : tensor<2560x2560xf32>) -> tensor<2560x2560xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2560, 2560], strides = [1, 1] : tensor<2560x2560xf32> -> !flow.dispatch.tensor<writeonly:tensor<2560x2560xf32>>
      return
    }
  }
}
}
