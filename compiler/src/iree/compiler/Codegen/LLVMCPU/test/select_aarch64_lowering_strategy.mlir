// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors  {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @matmul_tensors layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N}
        %init_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
              %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %init_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N} -> tensor<?x?xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [16, 4, 0], [0, 0, 64], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<Mmt4dTilingExpert>
//      CHECK: hal.executable.export public @matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors_sve  {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    cpu_features = "+sve",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @matmul_tensors layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N}
        %init_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
              %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %init_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N} -> tensor<?x?xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 128, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_static_tensors_sve  {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    cpu_features = "+sve",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @static_tensors_non_pow_two_sizes layout(#pipeline_layout)
    builtin.module {
      func.func @static_tensors_non_pow_two_sizes() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<15x14xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<14x7xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<15x7xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [15, 14], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<15x14xf32>> -> tensor<15x14xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [14, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<14x7xf32>> -> tensor<14x7xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [15, 7], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<15x7xf32>> -> tensor<15x7xf32>
        %6 = linalg.matmul ins(%3, %4 : tensor<15x14xf32>, tensor<14x7xf32>) outs(%5 : tensor<15x7xf32>) -> tensor<15x7xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [15, 7], strides = [1, 1] : tensor<15x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<15x7xf32>> return }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[5, 7, 0], [5, [8], 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @static_tensors_non_pow_two_sizes
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_static_tensors_sve  {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    cpu_features = "+sve",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @static_tensors_1x1 layout(#pipeline_layout)
    builtin.module {
      func.func @static_tensors_1x1() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>> -> tensor<1x1xf32>
        %6 = linalg.matmul ins(%3, %4 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%5 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @static_tensors_1x1
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @batch_matmul_tensors {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @batch_matmul_tensors layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %B = hal.interface.constant.load[0] : index
        %M = hal.interface.constant.load[1] : index
        %N = hal.interface.constant.load[2] : index
        %K = hal.interface.constant.load[3] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%B, %M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%B, %K, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
            : !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%B, %M, %N}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0, 0], sizes = [%B, %M, %K], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%B, %M, %K} -> tensor<?x?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0, 0], sizes = [%B, %K, %N], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%B, %K, %N} -> tensor<?x?x?xf32>
        %init = tensor.empty(%B, %M, %N) : tensor<?x?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        %batch_gemm = linalg.batch_matmul
            ins(%lhs, %rhs : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%fill : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
        flow.dispatch.tensor.store %batch_gemm, %result_binding, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
            : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%B, %M, %N}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 64, 0], [1, 8, 16, 0], [0, 0, 0, 1], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @batch_matmul_tensors
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
    hal.executable.export public @matmul_static layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_static() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<196x240xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<240x40xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<196x40xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [196, 240], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<196x240xf32>> -> tensor<196x240xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [240, 40], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<240x40xf32>> -> tensor<240x40xf32>
        %init = tensor.empty() : tensor<196x40xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<196x40xf32>) -> tensor<196x40xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<196x240xf32>, tensor<240x40xf32>)
            outs(%fill : tensor<196x40xf32>) -> tensor<196x40xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [196, 40], strides = [1, 1]
            : tensor<196x40xf32> -> !flow.dispatch.tensor<writeonly:tensor<196x40xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[28, 20, 0], [4, 8, 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//       CHECK: hal.executable.export public @matmul_static
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_static {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
    hal.executable.export public @conv_static layout(#pipeline_layout)
    builtin.module {
      func.func @conv_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c607520 = arith.constant 607520 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x51x41x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c607520) : !flow.dispatch.tensor<readonly:tensor<3x3x512x512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x25x20x512xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 51, 41, 512], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x51x41x512xf32>> -> tensor<1x51x41x512xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 512, 512], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x512x512xf32>> -> tensor<3x3x512x512xf32>
        %5 = tensor.empty() : tensor<1x25x20x512xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x25x20x512xf32>) -> tensor<1x25x20x512xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x51x41x512xf32>, tensor<3x3x512x512xf32>) outs(%6 : tensor<1x25x20x512xf32>) -> tensor<1x25x20x512xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 25, 20, 512], strides = [1, 1, 1, 1] : tensor<1x25x20x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x25x20x512xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 5, 20, 64, 0, 0, 0], [1, 1, 20, 64, 0, 0, 0], [0, 0, 0, 0, 1, 1, 16], [0, 0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @restrict_num_workgroups {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
    hal.executable.export public @restrict_num_workgroups layout(#pipeline_layout)
    builtin.module {
      func.func @restrict_num_workgroups() {
        %cst = arith.constant 0.000000e+00 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x7x7x576xf32>>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 11, 11, 576], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x11x11x576xf32>> -> tensor<1x11x11x576xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [5, 5, 576], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<5x5x576xf32>> -> tensor<5x5x576xf32>
        %init = tensor.empty() : tensor<1x7x7x576xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%input, %filter : tensor<1x11x11x576xf32>, tensor<5x5x576xf32>)
            outs(%fill : tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 7, 7, 576], strides = [1, 1, 1, 1]
            : tensor<1x7x7x576xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x7x7x576xf32>>
        return
      }
    }
  }
}
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 7, 7, 64, 0, 0], [1, 1, 1, 4, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//       CHECK: hal.executable.export public @restrict_num_workgroups
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.depthwise_conv_2d_nhwc_hwc
//  CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_aarch_i8_i8_i32_static  {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
  hal.executable.export public @matmul_aarch_i8_i8_i32_static layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_aarch_i8_i8_i32_static() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384x1536xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x1536xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xi8>> -> tensor<128x384xi8>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x1536xi8>> -> tensor<384x1536xi8>
        %5 = tensor.empty() : tensor<128x1536xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
        %7 = linalg.matmul ins(%3, %4 : tensor<128x384xi8>, tensor<384x1536xi8>) outs(%6 : tensor<128x1536xi32>) -> tensor<128x1536xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : tensor<128x1536xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x1536xi32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, 16, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @matmul_aarch_i8_i8_i32_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_aarch_i8_i8_i32_dynamic  {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
  hal.executable.export public @matmul_aarch_i8_i8_i32_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_aarch_i8_i8_i32_dynamic() {
        %c0 = arith.constant 0 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%K, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0)
            : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%M, %N}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%M, %K} -> tensor<?x?xi8>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xi8>>{%K, %N} -> tensor<?x?xi8>
        %init = flow.dispatch.tensor.load %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%M, %N} -> tensor<?x?xi32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xi8>, tensor<?x?xi8>) outs(%init : tensor<?x?xi32>) -> tensor<?x?xi32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%M, %N}
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, 16, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @matmul_aarch_i8_i8_i32_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @pack  {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
  hal.executable.export public @pack layout(#pipeline_layout)
    builtin.module {
      func.func @pack() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<20x40xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x48x8x1xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [20, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20x40xf32>> -> tensor<20x40xf32>
        %3 = tensor.empty() : tensor<4x48x8x1xf32>
        %4 = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<20x40xf32> -> tensor<4x48x8x1xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [4, 48, 8, 1], strides = [1, 1, 1, 1] : tensor<4x48x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x48x8x1xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 40], [1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: hal.executable.export public @pack
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @unpack_outer_dynamic  {
  hal.executable.variant public @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }>) {
  hal.executable.export public @unpack_outer_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @unpack_outer_dynamic() {
        %c131072 = arith.constant 131072 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = arith.index_castui %0 : i32 to index
        %5 = arith.index_castui %1 : i32 to index
        %6 = arith.index_castui %2 : i32 to index
        %7 = arith.index_castui %3 : i32 to index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c131072) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x32x16xi32>>{%4, %5} -> tensor<?x?x32x16xi32>
        %11 = tensor.empty(%6, %7) : tensor<?x?xi32>
        %12 = tensor.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %11 : tensor<?x?x32x16xi32> -> tensor<?x?xi32>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [32, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: hal.executable.export public @unpack_outer_dynamic
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 96)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 128)>
hal.executable private @mmt4d_384x384x512_4x1x4_dispatch_0 {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @mmt4d_384x384x512_4x1x4_dispatch_0 layout(#pipeline_layout)
    builtin.module  {
      func.func @mmt4d_384x384x512_4x1x4_dispatch_0() {
        %c0 = arith.constant 0 : index
        %c96 = arith.constant 96 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<96x384x4x1xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x384x4x1xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [96, 384, 4, 1], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<96x384x4x1xf32>> -> tensor<96x384x4x1xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, 384, 4, 1], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x384x4x1xf32>> -> tensor<128x384x4x1xf32>
        %11 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [96, 384, 4, 4], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>> -> tensor<96x128x4x4xf32>
        %12 = linalg.mmt4d
            ins(%8, %10 : tensor<96x384x4x1xf32>, tensor<128x384x4x1xf32>)
            outs(%11 : tensor<96x128x4x4xf32>) -> tensor<96x128x4x4xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [0, 0, 0, 0], sizes = [96, 128, 4, 4], strides = [1, 1, 1, 1]
            : tensor<96x128x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<96x128x4x4xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16, 0, 0, 0, 0], [1, 1, 0, 4, 4, 0], [0, 0, 1, 0, 0, 1]{{\]}}
//      CHECK: func.func @mmt4d_384x384x512_4x1x4_dispatch_0()
//      CHECK:   linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]
