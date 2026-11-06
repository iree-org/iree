// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-configuration-preprocessing-pipeline, builtin.module(iree-codegen-llvmgpu-configuration-pipeline, iree-codegen-llvmgpu-rocdl-lowering-pipeline), iree-codegen-translation-postprocessing-pipeline)))" %s \
// RUN: | FileCheck %s

// The Distribute fallback distributes only linalg ops, so a scan runs its
// serial body once: one thread of one workgroup.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @scan_rows {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @scan_rows ordinal(0) layout(#pipeline_layout)
        count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scan_rows() {
        %c0 = arith.constant 0 : index
        %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x10xi32>>
        %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6x10xi32>>
        %acc_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6xi32>>
        %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [6, 10], strides = [1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x10xi32>> -> tensor<6x10xi32>
        %zero = arith.constant 0 : i32
        %out_empty = tensor.empty() : tensor<6x10xi32>
        %acc_empty = tensor.empty() : tensor<6xi32>
        %acc_init = linalg.fill ins(%zero : i32) outs(%acc_empty : tensor<6xi32>) -> tensor<6xi32>
        %scan:2 = iree_linalg_ext.scan dimension(1) inclusive(true)
            ins(%input : tensor<6x10xi32>) outs(%out_empty, %acc_init : tensor<6x10xi32>, tensor<6xi32>) {
        ^bb0(%a: i32, %b: i32):
          %s = arith.addi %a, %b : i32
          iree_linalg_ext.yield %s : i32
        } -> tensor<6x10xi32>, tensor<6xi32>
        iree_tensor_ext.dispatch.tensor.store %scan#0, %out_binding, offsets = [0, 0], sizes = [6, 10], strides = [1, 1]
            : tensor<6x10xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6x10xi32>>
        iree_tensor_ext.dispatch.tensor.store %scan#1, %acc_binding, offsets = [0], sizes = [6], strides = [1]
            : tensor<6xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6xi32>>
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @scan_rows
//       CHECK:   hal.executable.export public @scan_rows
//       CHECK:     hal.return %[[C1:[a-z0-9_]+]], %[[C1]], %[[C1]]
//       CHECK:   workgroup_size = [1 : index, 1 : index, 1 : index]
//       CHECK:   llvm.func @scan_rows
//   CHECK-NOT:     rocdl.workitem.id
//   CHECK-NOT:     rocdl.workgroup.id
//       CHECK:     llvm.add
