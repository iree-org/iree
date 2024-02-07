// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SME
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file  --iree-experimental-llvmcpu-arm-force-ssve=true %s | FileCheck %s --check-prefixes=CHECK,SSVE-WITHOUT-SME

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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: hal.executable.export public @matmul_tensors
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[5, 7, 0], [5, [8], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: hal.executable.export public @static_tensors_non_pow_two_sizes
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @static_tensors_1x1  {
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

// -----

// CHECK-DAG:  #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [8, [16]], [0, 0], [0, 0]]>
// CHECK-DAG:  #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
// CHECK:      #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
// CHECK:  hal.executable.export public @matmul_with_fill
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG1]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

#pipeline_layout = #hal.pipeline.layout<push_constants = 5, sets = [
  <0, bindings = [
    <0, storage_buffer, ReadOnly>,
    <1, storage_buffer, ReadOnly>,
    <2, storage_buffer>
  ]>
]>
hal.executable private @matmul_with_fill {
  hal.executable.variant @llvm target(<"llvm-cpu", "system-elf-arm_64", {
    cpu = "",
    cpu_features = "+v9a,+sve",
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    link_embedded = false,
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android34"
  }>) {
    hal.executable.export @matmul_with_fill layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_with_fill() {
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = arith.index_castui %0 : i32 to index
        %6 = arith.index_castui %1 : i32 to index
        %7 = arith.index_castui %2 : i32 to index
        %8 = arith.index_castui %3 : i32 to index
        %9 = arith.index_castui %4 : i32 to index
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x256xi8>>
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
        %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024xf32>>
        %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
        %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%9) : !flow.dispatch.tensor<writeonly:tensor<1024x256xf32>>
        %15 = flow.dispatch.tensor.load %10, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x256xi8>> -> tensor<1024x256xi8>
        %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
        %17 = flow.dispatch.tensor.load %12, offsets = [0], sizes = [1024], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1024xf32>> -> tensor<1024xf32>
        %18 = flow.dispatch.tensor.load %13, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %19 = tensor.empty() : tensor<1024x256xf32>
        %20 = tensor.empty() : tensor<1024x256xi32>
        %21 = linalg.fill ins(%c0_i32 : i32) outs(%20 : tensor<1024x256xi32>) -> tensor<1024x256xi32>
        %22 = linalg.matmul ins(%15, %16 : tensor<1024x256xi8>, tensor<256x256xi8>) outs(%21 : tensor<1024x256xi32>) -> tensor<1024x256xi32>
        %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %17, %18 : tensor<1024x256xi32>, tensor<1024xf32>, tensor<256xf32>) outs(%19 : tensor<1024x256xf32>) {
        ^bb0(%in: i32, %in_0: f32, %in_1: f32, %out: f32):
          %24 = arith.sitofp %in : i32 to f32
          %25 = arith.mulf %24, %in_0 : f32
          %26 = arith.mulf %25, %in_1 : f32
          linalg.yield %26 : f32
        } -> tensor<1024x256xf32>
        flow.dispatch.tensor.store %23, %14, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : tensor<1024x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x256xf32>>
        return
      }
    }
  }
}

//  SSVE-WITHOUT-SME-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//  SSVE-WITHOUT-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      SSVE-WITHOUT-SME: hal.executable.export public @matmul_tensors
//  SSVE-WITHOUT-SME-SAME:     translation_info = #[[TRANSLATION]]
//       SSVE-WITHOUT-SME: linalg.matmul
//  SSVE-WITHOUT-SME-SAME:     lowering_config = #[[CONFIG]]

//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], {{\[}}[4], [4], 0], [0, 0, 1], [0, 0, 0]]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       WITH-SME: hal.executable.export public @matmul_tensors
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors_sme  {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    cpu_features = "+sve,+sme",
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
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export public @static_tensors_1x1
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
