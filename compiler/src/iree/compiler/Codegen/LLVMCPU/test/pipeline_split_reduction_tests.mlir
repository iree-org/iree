// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --iree-llvmcpu-reassociate-fp-reductions=false --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --iree-llvmcpu-reassociate-fp-reductions=true --split-input-file %s | FileCheck %s --check-prefix=REORDERCHECK

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @split_reduction_innermost_reduction_no_dynamic_perfect_tiling_supported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : tensor<1024x512xi32>
  %c1_i32 = arith.constant 1 : i32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 512, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xi32>> -> tensor<1024x512x256xi32>
  %3 = tensor.empty() : tensor<1024x512xi32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<1024x512x256xi32>) outs(%cst : tensor<1024x512xi32>) {
  ^bb0(%in: i32, %out: i32):
    %6 = arith.addi %in, %out : i32
    linalg.yield %6 : i32
  } -> tensor<1024x512xi32>
  %5 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1024x512xi32>) outs(%3 : tensor<1024x512xi32>) {
  ^bb0(%in: i32, %out: i32):
    %6 = arith.addi %in, %c1_i32 : i32
    linalg.yield %6 : i32
  } -> tensor<1024x512xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  return
}

// CHECK-LABEL: func.func @split_reduction_innermost_reduction_no_dynamic_perfect_tiling_supported()
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               scf.for %[[ARG3:.+]] = %[[C0]] to %[[C64]] step %[[C1]]
// CHECK:                 arith.addi
// CHECK:                 scf.yield %{{.*}} : vector<4xi32>
// CHECK:               vector.reduction <add>, %{{.+}} %{{.+}} : vector<4xi32> into i32
// CHECK:             arith.addi %{{.+}}, %{{.+}} : vector<4xi32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @split_reduction_innermost_reduction_no_dynamic_perfect_tiling_float_supported_with_flag() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<1024x512xf32>
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 512, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xf32>> -> tensor<1024x512x256xf32>
  %3 = tensor.empty() : tensor<1024x512xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<1024x512x256xf32>) outs(%cst : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1024x512xf32>
  %5 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1024x512xf32>) outs(%3 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %cst_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1024x512xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>>
  return
}

// CHECK-LABEL: func.func @split_reduction_innermost_reduction_no_dynamic_perfect_tiling_float_supported_with_flag()
// CHECK-NOT:     arith.addf : vector<4xf32>

// REORDERCHECK-LABEL: func.func @split_reduction_innermost_reduction_no_dynamic_perfect_tiling_float_supported_with_flag()
// REORDERCHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// REORDERCHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// REORDERCHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// REORDERCHECK:         scf.for
// REORDERCHECK:           scf.for
// REORDERCHECK:             scf.for
// REORDERCHECK:               scf.for %[[ARG3:.+]] = %[[C0]] to %[[C64]] step %[[C1]]
// REORDERCHECK:                 arith.addf
// REORDERCHECK:                 scf.yield %{{.*}} : vector<4xf32>
// REORDERCHECK:               vector.reduction <add>, %{{.+}} %{{.+}} : vector<4xf32> into f32
// REORDERCHECK:             arith.addf %{{.+}}, %{{.+}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @split_reduction_innermost_reduction_next_dynamic_supported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x?x256xi32>>{%1}
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x?xi32>>{%1}
  %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [1024, %1, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x?x256xi32>>{%1} -> tensor<1024x?x256xi32>
  %5 = tensor.empty(%1) : tensor<1024x?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<1024x?xi32>) -> tensor<1024x?xi32>
  %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4 : tensor<1024x?x256xi32>) outs(%6 : tensor<1024x?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %8 = arith.addi %in, %out : i32
    linalg.yield %8 : i32
  } -> tensor<1024x?xi32>
  iree_tensor_ext.dispatch.tensor.store %7, %3, offsets = [0, 0], sizes = [1024, %1], strides = [1, 1] : tensor<1024x?xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x?xi32>>{%1}
  return
}

// CHECK-LABEL:  func.func @split_reduction_innermost_reduction_next_dynamic_supported()
// CHECK-DAG:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:      %[[C64:.+]] = arith.constant 64 : index
// CHECK:          scf.for
// CHECK:            scf.for
// CHECK:              scf.for
// CHECK:                scf.for %[[ARG3:.+]] = %[[C0]] to %[[C64]] step %[[C1]]
// CHECK:                  arith.addi
// CHECK:                  scf.yield %{{.*}} : vector<4xi32>
// CHECK:                vector.reduction <add>, %{{.+}} %{{.+}} : vector<4xi32> into i32

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @split_reduction_innermost_reduction_next_imperfect_tiling_supported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : tensor<1024x513xi32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x513x256xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x513xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 513, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x513x256xi32>> -> tensor<1024x513x256xi32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<1024x513x256xi32>) outs(%cst : tensor<1024x513xi32>) {
  ^bb0(%in: i32, %out: i32):
    %4 = arith.addi %in, %out : i32
    linalg.yield %4 : i32
  } -> tensor<1024x513xi32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1024, 513], strides = [1, 1] : tensor<1024x513xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x513xi32>>
  return
}

// CHECK-LABEL:  func.func @split_reduction_innermost_reduction_next_imperfect_tiling_supported()
// CHECK-DAG:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:      %[[C64:.+]] = arith.constant 64 : index
// CHECK:          scf.for
// CHECK:            scf.for
// CHECK:              scf.for
// CHECK:                scf.for %[[ARG3:.+]] = %[[C0]] to %[[C64]] step %[[C1]]
// CHECK:                  arith.addi
// CHECK:                  scf.yield %{{.*}} : vector<4xi32>
// CHECK:                vector.reduction <add>, %{{.+}} %{{.+}} : vector<4xi32> into i32

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @split_reduction_innermost_dynamic_reduction_unsupported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant dense<0> : tensor<1024x512xi32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x?xi32>>{%1}
  %4 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [1024, 512, %1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x?xi32>>{%1} -> tensor<1024x512x?xi32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4 : tensor<1024x512x?xi32>) outs(%cst : tensor<1024x512xi32>) {
  ^bb0(%in: i32, %out: i32):
    %6 = arith.addi %in, %out : i32
    linalg.yield %6 : i32
  } -> tensor<1024x512xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  return
}

// CHECK-LABEL:  func.func @split_reduction_innermost_dynamic_reduction_unsupported()
//     CHECK-4:    vector.mask %{{.*}} { vector.reduction <add>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @split_reduction_innermost_imperfect_reduction_unsupported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : tensor<1024x512xi32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x257xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 512, 257], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x257xi32>> -> tensor<1024x512x257xi32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<1024x512x257xi32>) outs(%cst : tensor<1024x512xi32>) {
  ^bb0(%in: i32, %out: i32):
    %4 = arith.addi %in, %out : i32
    linalg.yield %4 : i32
  } -> tensor<1024x512xi32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xi32>>
  return
}

// CHECK-LABEL:  func.func @split_reduction_innermost_imperfect_reduction_unsupported()
//     CHECK-4:    vector.mask %{{.*}} { vector.reduction <add>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @split_reduction_not_innermost_reduction_unsupported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : tensor<1024x256xi32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x256xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 512, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xi32>> -> tensor<1024x512x256xi32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<1024x512x256xi32>) outs(%cst : tensor<1024x256xi32>) {
  ^bb0(%in: i32, %out: i32):
    %4 = arith.addi %in, %out : i32
    linalg.yield %4 : i32
  } -> tensor<1024x256xi32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : tensor<1024x256xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x256xi32>>
  return
}

// CHECK-LABEL:  func.func @split_reduction_not_innermost_reduction_unsupported()
// CHECK-NOT:      vector.reduction

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
func.func @split_reduction_double_reduction_unsupported() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1024, 512, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512x256xi32>> -> tensor<1024x512x256xi32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%2 : tensor<1024x512x256xi32>) outs(%cst : tensor<1024xi32>) {
  ^bb0(%in: i32, %out: i32):
    %4 = arith.addi %in, %out : i32
    linalg.yield %4 : i32
  } -> tensor<1024xi32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xi32>>
  return
}

// CHECK-LABEL:  func.func @split_reduction_double_reduction_unsupported()
// CHECK:          vector.insert %{{.+}}, %{{.+}} : i32 into vector<4xi32>
// CHECK-NOT:      vector.insert %{{.+}}, %{{.+}} : i32 into vector<1xi32>
