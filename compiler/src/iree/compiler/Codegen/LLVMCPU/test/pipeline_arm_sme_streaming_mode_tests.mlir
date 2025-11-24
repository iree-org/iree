// RUN: iree-opt --iree-codegen-linalg-to-llvm-pipeline=enable-arm-sme --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-codegen-linalg-to-llvm-pipeline=enable-arm-sme --iree-llvmcpu-force-arm-streaming --split-input-file %s | FileCheck %s -check-prefixes=FORCE-ARM-STREAMING

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_cpu.lowering_config<distribution = [0], vector_common_parallel = [1]>
func.func @fixed_size_dispatch() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xf32>>
  %1 = tensor.empty() : tensor<1xf32>
  %2 = linalg.fill {lowering_config = #config}
      ins(%cst : f32) outs(%1 : tensor<1xf32>) -> tensor<1xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xf32>>
  return
}

/// A dispatch region that only uses fixed-size vectors should never use
/// streaming mode.

// CHECK: @fixed_size_dispatch
// CHECK-NOT: arm_locally_streaming
// CHECK-NOT: arm_new_za

// FORCE-ARM-STREAMING: @fixed_size_dispatch
// FORCE-ARM-STREAMING-NOT: arm_locally_streaming
// FORCE-ARM-STREAMING-NOT: arm_new_za

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config1 = #iree_cpu.lowering_config<distribution = [0], vector_common_parallel = [[1]]>
func.func @scalable_dispatch() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xf32>>
  %1 = tensor.empty() : tensor<1xf32>
  %2 = linalg.fill {lowering_config = #config1}
      ins(%cst : f32) outs(%1 : tensor<1xf32>) -> tensor<1xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xf32>>
  return
}

/// A dispatch region that uses scalable vectors (but not ArmSME dialect
/// operations) should only use streaming if `--iree-llvmcpu-force-arm-streaming`
/// is set.

// CHECK: @scalable_dispatch
// CHECK-NOT: arm_locally_streaming
// CHECK-NOT: arm_new_za

// FORCE-ARM-STREAMING: @scalable_dispatch
// FORCE-ARM-STREAMING-NOT: arm_new_za
// FORCE-ARM-STREAMING-SAME: arm_locally_streaming

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config2 = #iree_cpu.lowering_config<distribution = [0, 0], vector_common_parallel = [[4], [4]]>
func.func @scalable_dispatch_using_za() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>,
    translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<100x100xf32>>
  %1 = tensor.empty() : tensor<100x100xf32>
  %2 = linalg.fill {lowering_config = #config2}
      ins(%cst : f32) outs(%1 : tensor<100x100xf32>) -> tensor<100x100xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [100, 100], strides = [1, 1] : tensor<100x100xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<100x100xf32>>
  return
}

/// A dispatch region that uses ArmSME operations (that require the ZA state)
/// should ways have streaming mode and ZA enabled.

// CHECK: @scalable_dispatch_using_za
// CHECK-SAME: arm_locally_streaming
// CHECK-SAME: arm_new_za

// FORCE-ARM-STREAMING: @scalable_dispatch_using_za
// FORCE-ARM-STREAMING-SAME: arm_locally_streaming
// FORCE-ARM-STREAMING-SAME: arm_new_za
