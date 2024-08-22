// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' -split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @mask_dynamic_generic_add() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %9 = tensor.empty(%2, %3) : tensor<?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  return
}

// Masking is applied to the main vector loop when peeling is not used.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
//         CHECK: scf.for
// CHECK-COUNT-2:   vector.maskedload
//         CHECK:   vector.maskedstore
// No epilogue
//     CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @mask_dynamic_reduction() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%2}
  %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %7 = tensor.empty(%2) : tensor<?xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?xf32>) -> tensor<?xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<?x?xf32>) outs(%8 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %10 = arith.addf %out, %in : f32
    linalg.yield %10 : f32
  } -> tensor<?xf32>
  flow.dispatch.tensor.store %9, %5, offsets = [0], sizes = [%2], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%2}
  return
}

//   CHECK-LABEL: func.func @mask_dynamic_reduction
//         CHECK:   vector.maskedload
//         CHECK:   vector.mask %{{.*}} { vector.reduction <add>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_riscv_32_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_32", {data_layout = "e-m:e-p:32:32-i64:64-n32-S128", native_vector_size = 32 : index, target_triple = "riscv32-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @mask_dynamic_generic_add() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_32_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %9 = tensor.empty(%2, %3) : tensor<?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  return
}

// Masking is applied to the main vector loop when peeling is not used.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
//         CHECK: scf.for
// CHECK-COUNT-2:   vector.maskedload
//         CHECK:   vector.maskedstore
// No epilogue
//     CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @mask_dynamic_generic_add() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %9 = tensor.empty(%2, %3) : tensor<?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  return
}

// Masking should not happen on aarch64 if there is no SVE support.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
//   CHECK-NOT:   vector.maskedload

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @mask_matmul_sve() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  %7 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
  %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  flow.dispatch.tensor.store %10, %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
  return
}

// Masking is applied to the matmul on aarch64 when SVE is enabled.

// CHECK-LABEL: func.func @mask_matmul_sve
// CHECK:   vector.maskedload

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @mask_dynamic_generic_add() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
  %9 = tensor.empty(%2, %3) : tensor<?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%2, %3}
  return
}

// Masking is applied to the peeled loop on aarch64 when SVE is enabled.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
//         CHECK: scf.for
// Peeled loop:
//         CHECK: scf.for
// CHECK-COUNT-2:   vector.maskedload
//         CHECK:   vector.maskedstore
