// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx90a --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-rocdl-select-lowering-strategy, func.func(iree-rocdl-lower-executable-target)))))' -mlir-print-local-scope %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable @scalar_dispatch {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @scalar_dispatch ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @scalar_dispatch() {
        %c0 = arith.constant 0 : index
        %c6364136223846793005_i64 = arith.constant 6364136223846793005 : i64
        %c1442695040888963407_i64 = arith.constant 1442695040888963407 : i64
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i64>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
        %extracted = tensor.extract %2[] : tensor<i64>
        %3 = arith.muli %extracted, %c6364136223846793005_i64 : i64
        %4 = arith.addi %3, %c1442695040888963407_i64 : i64
        %inserted = tensor.insert %4 into %2[] : tensor<i64>
        iree_tensor_ext.dispatch.tensor.store %inserted, %1, offsets = [], sizes = [], strides = [] : tensor<i64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i64>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @scalar_dispatch()
//  CHECK-SAME: translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUBaseLowering workgroup_size = [1, 1, 1]>
//       CHECK:   %[[SPANBIND0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//       CHECK:   %[[ASSUMED_SPAN0:.+]] = memref.assume_alignment %[[SPANBIND0]], 64
//       CHECK:   %[[SPAN0:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_SPAN0]]
//       CHECK:   %[[SPANBIND1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK:   %[[ASSUMED_SPAN1:.+]] = memref.assume_alignment %[[SPANBIND1]], 64
//       CHECK:   %[[SPAN1:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_SPAN1]]
//       CHECK:   memref.load %[[SPAN0]][] : memref<i64, #amdgpu.address_space<fat_raw_buffer>>
//       CHECK:   arith.muli {{.+}} : i64
//       CHECK:   arith.addi {{.+}} : i64
//       CHECK:   memref.store %{{.+}}, %[[SPAN1]][] : memref<i64, #amdgpu.address_space<fat_raw_buffer>>
