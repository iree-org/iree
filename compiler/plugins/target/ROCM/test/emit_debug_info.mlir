// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --mlir-print-debuginfo --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline{preserve-debug-info})))" %s | FileCheck %s --check-prefix=WITH-DEBUG
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --mlir-print-debuginfo --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=WITHOUT-DEBUG

// Test that debug location information is preserved when the pipeline
// `preserve-debug-info` option is used, and stripped otherwise.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable @debug_test {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @debug_test layout(#pipeline_layout)
    builtin.module {
      func.func @debug_test() {
        %c0 = arith.constant 0 : index loc("test.mlir":1:1)
        %c2 = arith.constant 2.0 : f32 loc("test.mlir":2:1)
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> loc("test.mlir":3:1)
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>> loc("test.mlir":4:1)
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[4], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32> loc("test.mlir":5:1)
        %3 = tensor.empty() : tensor<4xf32> loc("test.mlir":6:1)
        %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
             ins(%2 : tensor<4xf32>) outs(%3 : tensor<4xf32>) {
        ^bb0(%in: f32 loc("test.mlir":8:1), %out: f32 loc("test.mlir":8:2)):
          %5 = arith.mulf %in, %c2 : f32 loc("test.mlir":9:1)
          linalg.yield %5 : f32 loc("test.mlir":10:1)
        } -> tensor<4xf32> loc("test.mlir":7:1)
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets=[0], sizes=[4], strides=[1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>> loc("test.mlir":11:1)
        return loc("test.mlir":12:1)
      }
    }
  }
}

// WITH-DEBUG: loc("test.mlir":9:1
// WITHOUT-DEBUG-NOT: loc("test.mlir":
