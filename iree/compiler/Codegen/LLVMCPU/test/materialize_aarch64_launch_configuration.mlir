// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -split-input-file %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors  {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.entry_point @matmul_tensors layout(#executable_layout)
    builtin.module {
      func.func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%K, %N}
        %init_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:?x?xf32>{%M, %N}
              %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%K, %N} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %init_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %N} -> tensor<?x?xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%M, %N}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [16, 4, 64], [4, 4, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUTileFuseAndVectorize>
//      CHECK: hal.executable.entry_point public @matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @batch_matmul_tensors {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.entry_point @batch_matmul_tensors layout(#executable_layout)
    builtin.module {
      func.func @batch_matmul_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %B = hal.interface.constant.load[0] : index
        %M = hal.interface.constant.load[1] : index
        %N = hal.interface.constant.load[2] : index
        %K = hal.interface.constant.load[3] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %K, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:?x?x?xf32>{%B, %M, %N}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0, 0], sizes = [%B, %M, %K], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %M, %K} -> tensor<?x?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0, 0], sizes = [%B, %K, %N], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:?x?x?xf32>{%B, %K, %N} -> tensor<?x?x?xf32>
        %init = linalg.init_tensor [%B, %M, %N] : tensor<?x?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %batch_gemm = linalg.batch_matmul
            ins(%lhs, %rhs : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%fill : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        flow.dispatch.tensor.store %batch_gemm, %result_binding, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
            : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?x?xf32>{%B, %M, %N}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64, 0], [1, 16, 4, 64], [1, 4, 4, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUTileFuseAndVectorize>
//      CHECK: hal.executable.entry_point public @batch_matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @matmul_static layout(#executable_layout)
    builtin.module {
      func.func @matmul_static() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:196x240xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:240x40xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:196x40xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [196, 240], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:196x240xf32> -> tensor<196x240xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [240, 40], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:240x40xf32> -> tensor<240x40xf32>
        %init = linalg.init_tensor [196, 40] : tensor<196x40xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<196x40xf32>) -> tensor<196x40xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<196x240xf32>, tensor<240x40xf32>)
            outs(%fill : tensor<196x40xf32>) -> tensor<196x40xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [196, 40], strides = [1, 1]
            : tensor<196x40xf32> -> !flow.dispatch.tensor<writeonly:196x40xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[49, 8, 0], [7, 4, 60], [4, 4, 4]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUTileFuseAndVectorize>
//       CHECK: hal.executable.entry_point public @matmul_static
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @restrict_num_workgroups {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @restrict_num_workgroups layout(#executable_layout)
    builtin.module {
      func.func @restrict_num_workgroups() {
        %cst = arith.constant 0.000000e+00 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:1x11x11x576xf32>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:5x5x576xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:1x7x7x576xf32>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 11, 11, 576], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x11x11x576xf32> -> tensor<1x11x11x576xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [5, 5, 576], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:5x5x576xf32> -> tensor<5x5x576xf32>
        %init = linalg.init_tensor [1, 7, 7, 576] : tensor<1x7x7x576xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %filter : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>)
            outs(%fill : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 7, 7, 576], strides = [1, 1, 1, 1]
            : tensor<1x7x7x576xf32> -> !flow.dispatch.tensor<writeonly:1x7x7x576xf32>
        return
      }
    }
  }
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 7, 64, 0, 0], [1, 1, 7, 8, 0, 0], [0, 0, 0, 0, 1, 1]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//       CHECK: hal.executable.entry_point public @restrict_num_workgroups
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.depthwise_conv_2d_nhwc_hwc
//  CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_aarch_i8_i8_i32  {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
  hal.executable.entry_point public @matmul_aarch_i8_i8_i32 layout(#executable_layout)
    builtin.module {
      func.func @matmul_aarch_i8_i8_i32() {
        %c0 = arith.constant 0 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?xi8>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readonly:?x?xi8>{%K, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32)
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%M, %N}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi8>{%M, %K} -> tensor<?x?xi8>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xi8>{%K, %N} -> tensor<?x?xi8>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:?x?xi32>{%M, %N} -> tensor<?x?xi32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xi8>, tensor<?x?xi8>) outs(%init : tensor<?x?xi32>) -> tensor<?x?xi32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>{%M, %N}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [4, 16, 0], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @matmul_aarch_i8_i8_i32
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]
