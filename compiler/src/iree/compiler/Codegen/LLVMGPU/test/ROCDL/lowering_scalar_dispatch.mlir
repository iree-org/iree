// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx90a --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-rocdl-select-lowering-strategy, func.func(iree-rocdl-lower-executable-target)))))' -mlir-print-local-scope %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

hal.executable @scalar_dispatch {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @scalar_dispatch ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @scalar_dispatch() {
        %c0 = arith.constant 0 : index
        %c6364136223846793005_i64 = arith.constant 6364136223846793005 : i64
        %c1442695040888963407_i64 = arith.constant 1442695040888963407 : i64
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<i64>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<i64>>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
        %extracted = tensor.extract %2[] : tensor<i64>
        %3 = arith.muli %extracted, %c6364136223846793005_i64 : i64
        %4 = arith.addi %3, %c1442695040888963407_i64 : i64
        %inserted = tensor.insert %4 into %2[] : tensor<i64>
        flow.dispatch.tensor.store %inserted, %1, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @scalar_dispatch()
//  CHECK-SAME: translation_info = #iree_codegen.translation_info<LLVMGPUBaseLowering workgroup_size = [1, 1, 1]>
//       CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(0)
//       CHECK:   %[[SPAN1:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(1)
//       CHECK:   memref.load %[[SPAN0]][] : memref<i64, #hal.descriptor_type<storage_buffer>>
//       CHECK:   arith.muli {{.+}} : i64
//       CHECK:   arith.addi {{.+}} : i64
//       CHECK:   memref.store %{{.+}}, %[[SPAN1]][] : memref<i64, #hal.descriptor_type<storage_buffer>>
