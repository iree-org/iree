// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' \
// RUN: --iree-llvmcpu-vector-pproc-strategy=peel --split-input-file %s | FileCheck %s

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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
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

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[5, 7, 0], [5, 7, 0], [0, 0, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
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

// TODO: FIXME - scalable "16" ([16]) for just 1 element
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, [16], 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @static_tensors_1x1
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
