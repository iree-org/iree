// RUN: iree-opt --iree-codegen-linalg-to-llvm-pipeline=enable-arm-sme-lowering-pipeline --split-input-file %s | FileCheck %s
// RUN: iree-opt --iree-codegen-linalg-to-llvm-pipeline=enable-arm-sme-lowering-pipeline --iree-llvmcpu-force-arm-streaming --split-input-file %s | FileCheck %s -check-prefixes=FORCE-ARM-STREAMING

module {
module {
  func.func @fixed_size_dispatch() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>}
        ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    return
  }
}
}

/// A dispatch region that only uses fixed-size vectors should never use
/// streaming mode.

// CHECK: @fixed_size_dispatch
// CHECK-NOT: arm_locally_streaming

// FORCE-ARM-STREAMING: @fixed_size_dispatch
// FORCE-ARM-STREAMING-NOT: arm_locally_streaming

// -----

module {
module {
  func.func @scalable_dispatch() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [[1]], [0], [0]]>}
        ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    return
  }
}
}

/// A dispatch region that uses scalable vectors (but not ArmSME dialect
/// operations) should only use streaming if `--iree-llvmcpu-force-arm-streaming`
/// is set.

// CHECK: @scalable_dispatch
// CHECK-NOT: arm_locally_streaming

// FORCE-ARM-STREAMING: @scalable_dispatch
// FORCE-ARM-STREAMING-SAME: arm_locally_streaming

// -----

module {
module {
  func.func @scalable_dispatch_using_za() attributes {hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<100x100xf32>>
    %2 = tensor.empty() : tensor<100x100xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [[4], [4]], [0, 0], [0, 0]]>}
        ins(%cst : f32) outs(%2 : tensor<100x100xf32>) -> tensor<100x100xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [100, 100], strides = [100, 1] : tensor<100x100xf32> -> !flow.dispatch.tensor<readwrite:tensor<100x100xf32>>
    return
  }
}
}

/// A dispatch region that uses ArmSME operations (that require the ZA state)
/// should ways have streaming mode and ZA enabled.

// CHECK: @scalable_dispatch_using_za
// CHECK-SAME: arm_locally_streaming
// CHECK-SAME: arm_new_za

// FORCE-ARM-STREAMING: @scalable_dispatch_using_za
// FORCE-ARM-STREAMING-SAME: arm_locally_streaming
// FORCE-ARM-STREAMING-SAME: arm_new_za
