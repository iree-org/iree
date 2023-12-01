// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-linalg-ext-decompose-softmax)), iree-llvmgpu-select-lowering-strategy,  iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @softmax {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>) {
  hal.executable.export @softmax layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @softmax() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -3.40282347E+38 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>> -> tensor<12x128x40960xf32>
      %3 = tensor.empty() : tensor<12x128x40960xf32>
      %4 = linalg.softmax dimension(2) ins(%2 : tensor<12x128x40960xf32>) outs(%3 : tensor<12x128x40960xf32>) -> tensor<12x128x40960xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      return
    }
  }
}
}

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @softmax
// CHECK-SAME:    subgroup_size = 32
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-SAME:    workgroup_size = [1024 : index, 1 : index, 1 : index]
// CHECK-LABEL:     func.func @softmax
// CHECK-COUNT-20:     gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @softmax {
hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx940"}>) {
  hal.executable.export @softmax layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @softmax() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -3.40282347E+38 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>> -> tensor<12x128x40960xf32>
      %3 = tensor.empty() : tensor<12x128x40960xf32>
      %4 = linalg.softmax dimension(2) ins(%2 : tensor<12x128x40960xf32>) outs(%3 : tensor<12x128x40960xf32>) -> tensor<12x128x40960xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      return
    }
  }
}
}

// On CDNA, we prefer wave64 with subgroup size 64.

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @softmax
// CHECK-SAME:    subgroup_size = 64
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-SAME:    workgroup_size = [1024 : index, 1 : index, 1 : index]
// CHECK-LABEL:     func.func @softmax
// CHECK-COUNT-20:     gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dynamic_softmax {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>) {
    hal.executable.export public @dynamic_softmax ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dynamic_softmax() {
        %c32_i64 = arith.constant 32 : i64
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.extui %0 : i32 to i64
        %3 = arith.extui %1 : i32 to i64
        %4 = arith.shli %3, %c32_i64 : i64
        %5 = arith.ori %2, %4 : i64
        %6 = arith.index_castui %5 : i64 to index
        %7 = flow.dispatch.workload.ordinal %6, 0 : index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?xf16>>{%7}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x?xf16>>{%7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [32, %7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?xf16>>{%7} -> tensor<32x?xf16>
        %11 = tensor.empty(%7) : tensor<32x?xf16>
        %12 = linalg.softmax dimension(1) ins(%10 : tensor<32x?xf16>) outs(%11 : tensor<32x?xf16>) -> tensor<32x?xf16>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [32, %7], strides = [1, 1] : tensor<32x?xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x?xf16>>{%7}
        return
      }
    }
  }
}

// Finer details of this lowering are captured by the spirv pipeline test. Just
// verify that warp reduction triggers.
// CHECK-LABEL: func.func @dynamic_softmax
// CHECK-COUNT-10: gpu.shuffle  xor {{.*}} : i32
