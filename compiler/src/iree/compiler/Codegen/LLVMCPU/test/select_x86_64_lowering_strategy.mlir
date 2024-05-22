// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @matvec_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
    %5 = tensor.empty() : tensor<128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
    %7 = linalg.matvec ins(%3, %4 : tensor<128x384xf32>, tensor<384xf32>) outs(%6 : tensor<128xf32>) -> tensor<128xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0], sizes = [128], strides = [1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 0], [32, 0], [0, 0], [32, 0], [0, 16], [0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//       CHECK: func.func @matvec_static()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matvec
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @matvec_dynamic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = arith.index_cast %0 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = arith.index_cast %2 : i32 to index
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %4}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%5}
    %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%3}
    %9 = hal.interface.constant.load[0] : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [%10], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%3} -> tensor<?xf32>
    %12 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%3, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%3, %4} -> tensor<?x?xf32>
    %13 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%5} -> tensor<?xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
    %15 = linalg.matvec ins(%12, %13 : tensor<?x?xf32>, tensor<?xf32>) outs(%14 : tensor<?xf32>) -> tensor<?xf32>
    flow.dispatch.tensor.store %15, %8, offsets = [0], sizes = [%3], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%3}
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 0], [64, 0], [0, 0], [32, 0], [0, 16], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matvec_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matvec
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @dot_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
    %5 = tensor.empty() : tensor<f32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
    %7 = linalg.dot ins(%3, %4 : tensor<384xf32>, tensor<384xf32>) outs(%6 : tensor<f32>) -> tensor<f32>
    flow.dispatch.tensor.store %7, %2, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [0], [0], [16], [0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @dot_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.dot
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @dot_dynamic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
    %5 = flow.dispatch.tensor.load %4, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<writeonly:tensor<f32>> -> tensor<f32>
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3}
    %8 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [%2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2} -> tensor<?xf32>
    %9 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [%3], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3} -> tensor<?xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%5 : tensor<f32>) -> tensor<f32>
    %11 = linalg.dot ins(%8, %9 : tensor<?xf32>, tensor<?xf32>) outs(%10 : tensor<f32>) -> tensor<f32>
    flow.dispatch.tensor.store %11, %4, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [0], [0], [16], [0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @dot_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.dot
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @dynamic_add() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1}
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %6 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [%1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1} -> tensor<?xf32>
    %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : tensor<?x?xf32>, tensor<?xf32>) outs(%7 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.addf %in, %in_0 : f32
      linalg.yield %9 : f32
    } -> tensor<?x?xf32>
    flow.dispatch.tensor.store %8, %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [1, 4], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @dynamic_add()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @add4D() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%9 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.addf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<?x?x?x?xf32>
    flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 64, 64], [1, 1, 1, 4], [0, 0, 0, 0], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @add4D()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @add_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x16x32x128xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x16x32x128xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 16, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x16x32x128xf32>> -> tensor<64x16x32x128xf32>
    %3 = tensor.empty() : tensor<64x16x32x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<64x16x32x128xf32>) outs(%3 : tensor<64x16x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<64x16x32x128xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [64, 16, 32, 128], strides = [1, 1, 1, 1] : tensor<64x16x32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x16x32x128xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 32, 64], [1, 1, 1, 4], [0, 0, 0, 0], [0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @add_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [64, 64, 0], [0, 0, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert, {enable_loop_peeling}>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @preset_config_matmul_tensors() attributes {
      hal.executable.target = #executable_target_system_elf_x86_64_,
      translation_info = #translation
    } {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
    %5 = tensor.empty() : tensor<128x512xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @preset_config_matmul_tensors()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @matmul_partially_peel() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16641x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<16x8xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16641x8xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16641, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16641x16xf32>> -> tensor<16641x16xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x8xf32>> -> tensor<16x8xf32>
    %5 = tensor.empty() : tensor<16641x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16641x8xf32>) -> tensor<16641x8xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<16641x16xf32>, tensor<16x8xf32>) outs(%6 : tensor<16641x8xf32>) -> tensor<16641x8xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [16641, 8], strides = [1, 1] : tensor<16641x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<16641x8xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[43, 8, 0], [43, 8, 0], [0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_partially_peel()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @copy_op_dynamic() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32>{%0, %1}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32>{%2, %3}
    %subview = memref.subview %7[%4, %5] [%0, %1] [1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<?x?xi32>) outs(%subview : memref<?x?xi32, strided<[?, 1], offset: ?>>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [1, 4], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
//      CHECK: func.func @copy_op_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @static_1d_fft_stage2() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
    %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
    %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
    %4:2 = iree_linalg_ext.fft ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
    flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//       CHECK: func.func @static_1d_fft_stage2()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @static_3d_fft_stage3() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %c3 = arith.constant 3 : index
    %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
    %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
    %0 = bufferization.to_memref %cst_0 : memref<4xf32>
    %1 = bufferization.to_memref %cst : memref<4xf32>
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x128x32xf32>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x128x32xf32>
    iree_linalg_ext.fft ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>) outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 64, 64]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//       CHECK: func.func @static_3d_fft_stage3()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @outs_fusion_fn() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %6 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %7 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
    %9 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%7 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.mulf %in, %in_0 : f32
      %12 = arith.addf %11, %out : f32
      linalg.yield %11 : f32
    } -> tensor<?x?xf32>
    flow.dispatch.tensor.store %10, %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32], [1, 4], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32, 0], [1, 4, 0], [0, 0, 4], [0, 0, 0]]>
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @outs_fusion_fn()
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//  CHECK-SAME:    lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @conv_dynamic() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.constant.load[6] : index
    %7 = hal.interface.constant.load[7] : index
    %8 = hal.interface.constant.load[8] : index
    %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%4, %5, %3, %6}
    %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %7, %8, %6}
    %12 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %13 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0, 0], sizes = [%4, %5, %3, %6], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%4, %5, %3, %6} -> tensor<?x?x?x?xf32>
    %14 = flow.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [%0, %7, %8, %6], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %7, %8, %6} -> tensor<?x?x?x?xf32>
    %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%12, %13 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%14 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    flow.dispatch.tensor.store %15, %11, offsets = [0, 0, 0, 0], sizes = [%0, %7, %8, %6], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xf32>>{%0, %7, %8, %6}
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 64, 64, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//      CHECK:         lowering_config = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @conv_static() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c607520 = arith.constant 607520 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c607520) : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
    %5 = tensor.empty() : tensor<1x112x112x16xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 16, 56, 16, 0, 0, 0], [1, 1, 4, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 3], [0, 0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @conv_nchw_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x128x30x30xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x128x3x3xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x128x28x28xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 128, 30, 30], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x128x30x30xf32>> -> tensor<1x128x30x30xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, 128, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x128x3x3xf32>> -> tensor<128x128x3x3xf32>
    %5 = tensor.empty() : tensor<1x128x28x28xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%6 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 128, 28, 28], strides = [1, 1, 1, 1] : tensor<1x128x28x28xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x128x28x28xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 64, 28, 4, 0, 0, 0], [1, 4, 1, 4, 0, 0, 0], [0, 0, 0, 0, 8, 1, 1], [0, 0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @conv_nchw_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.conv_2d_nchw_fchw

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @depthwise_conv_static() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x161x161x240xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x240xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x80x80x240xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x161x161x240xf32>> -> tensor<1x161x161x240xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x240xf32>> -> tensor<3x3x240xf32>
    %5 = tensor.empty() : tensor<1x80x80x240xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x80x80x240xf32>) -> tensor<1x80x80x240xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x161x161x240xf32>, tensor<3x3x240xf32>) outs(%6 : tensor<1x80x80x240xf32>) -> tensor<1x80x80x240xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 80, 80, 240], strides = [1, 1, 1, 1] : tensor<1x80x80x240xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x80x80x240xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 40, 40, 16, 0, 0], [1, 1, 4, 16, 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @depthwise_conv_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-linux-gnu"}>
module {
  func.func @thin_depthwise_conv_static() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>> -> tensor<1x57x57x72xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>> -> tensor<3x3x72xf32>
    %5 = tensor.empty() : tensor<1x28x28x72xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>) outs(%6 : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 72], strides = [1, 1, 1, 1] : tensor<1x28x28x72xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 28, 36, 0, 0], [1, 1, 7, 12, 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @thin_depthwise_conv_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+fma,+avx512f,+bmi,+bmi2,+aes,+pclmul,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vnni,+adx,+clflushopt,+clwb,+cx16,+cx8,+crc32,+f16c,+fsgsbase,+fxsr,+invpcid,+lzcnt,+movbe,+pku,+prfchw,+rdrnd,+rdseed,+sahf,+x87,+xsave,+xsavec,+xsaveopt,+xsaves", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf", ukernels = false}>
module {
  func.func @pooling_nchw_max() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c3846080 = arith.constant 3846080 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant -3.40282347E+38 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c3846080) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x64x114x114xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x64x56x56xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 64, 114, 114], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x64x114x114xf32>> -> tensor<1x64x114x114xf32>
    %3 = tensor.empty() : tensor<1x64x56x56xf32>
    %4 = tensor.empty() : tensor<3x3xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %6 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%2, %4 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0, 0], sizes = [1, 64, 56, 56], strides = [1, 1, 1, 1] : tensor<1x64x56x56xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x64x56x56xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 32, 56, 8, 0, 0], [1, 8, 1, 8, 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: func.func @pooling_nchw_max()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.pooling_nchw_max
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-pc-linux-gnu"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @generic_static() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<96x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<16x96xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [96, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<96x16xf32>> -> tensor<96x16xf32>
    %3 = tensor.empty() : tensor<16x96xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<96x16xf32>) outs(%3 : tensor<16x96xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x96xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [16, 96], strides = [1, 1] : tensor<16x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x96xf32>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 96], [16, 16], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @generic_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
//      CHECK:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @matmul_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
    %5 = tensor.empty() : tensor<384x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [384, 128], strides = [1, 1] : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[48, 64, 0], [48, 64, 0], [0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func @predict_dispatch_86(%arg0: !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<7xf32>>) attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+01 : f32
    %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [7, 7, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x2048xf32>> -> tensor<7x7x2048xf32>
    %1 = tensor.empty() : tensor<7xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<7xf32>) -> tensor<7xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%0 : tensor<7x7x2048xf32>) outs(%2 : tensor<7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<7xf32>
    %4 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%3 : tensor<7xf32>) outs(%1 : tensor<7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.divf %in, %cst_0 : f32
      linalg.yield %5 : f32
    } -> tensor<7xf32>
    flow.dispatch.tensor.store %4, %arg1, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:tensor<7xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 0, 0], [1, 0, 0], [0, 1, 4], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @predict_dispatch_86(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic {indexing_maps = [#{{.+}}, #{{.+}}], iterator_types = ["parallel", "reduction", "reduction"]}
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @matmul_i8_i8_i32_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [64, 64, 0], [0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_i8_i8_i32_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @gemm_unit_N() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%1}
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%0}
    %5 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%1} -> tensor<?x1xf32>
    %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%0} -> tensor<?x1xf32>
    %8 = linalg.matmul ins(%6, %5 : tensor<?x?xf32>, tensor<?x1xf32>) outs(%7 : tensor<?x1xf32>) -> tensor<?x1xf32>
    flow.dispatch.tensor.store %8, %4, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1] : tensor<?x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x1xf32>>{%0}
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 0, 0], [64, 0, 0], [0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @gemm_unit_N()
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @gemm_unit_M_unit_N() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0}
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%0}
    %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0} -> tensor<1x?xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1xf32>>{%0} -> tensor<?x1xf32>
    %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x1xf32>> -> tensor<1x1xf32>
    %7 = linalg.matmul ins(%4, %5 : tensor<1x?xf32>, tensor<?x1xf32>) outs(%6 : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.dispatch.tensor.store %7, %3, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @gemm_unit_M_unit_N()
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @matmul_odd() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<33x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x49xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<33x49xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<33x49xf32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x16xf32>> -> tensor<33x16xf32>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x49xf32>> -> tensor<16x49xf32>
    %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [33, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x49xf32>> -> tensor<33x49xf32>
    %7 = tensor.empty() : tensor<33x49xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<33x49xf32>) -> tensor<33x49xf32>
    %9 = linalg.matmul ins(%4, %5 : tensor<33x16xf32>, tensor<16x49xf32>) outs(%8 : tensor<33x49xf32>) -> tensor<33x49xf32>
    flow.dispatch.tensor.store %9, %3, offsets = [0, 0], sizes = [33, 49], strides = [1, 1] : tensor<33x49xf32> -> !flow.dispatch.tensor<writeonly:tensor<33x49xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 49, 0], [8, 49, 0], [0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_odd()
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
module {
  func.func @generic_unit_dims_dynamic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
    %6 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, %0, 1, 1, %1, %2, 1, %3], strides = [1, 1, 1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3} -> tensor<1x?x1x1x?x?x1x?xf32>
    %7 = tensor.empty(%0, %1, %2, %3) : tensor<1x?x1x1x?x?x1x?xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x?x1x1x?x?x1x?xf32>) outs(%7 : tensor<1x?x1x1x?x?x1x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %in : f32
      linalg.yield %9 : f32
    } -> tensor<1x?x1x1x?x?x1x?xf32>
    flow.dispatch.tensor.store %8, %5, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, %0, 1, 1, %1, %2, 1, %3], strides = [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x1x1x?x?x1x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x?x1x1x?x?x1x?xf32>>{%0, %1, %2, %3}
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0, 0, 64, 64, 0, 64], [1, 1, 1, 1, 1, 1, 1, 4], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @generic_unit_dims_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @reduce_to_scalar_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%2 : tensor<128xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [4], [0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @reduce_to_scalar_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @reduce_to_scalar_dynamic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0}
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<f32>>
    %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [%0], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
    %4 = flow.dispatch.tensor.load %2, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:tensor<f32>> -> tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%3 : tensor<?xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    flow.dispatch.tensor.store %5, %2, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<readwrite:tensor<f32>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0], [0], [4], [0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @reduce_to_scalar_dynamic()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#map = affine_map<() -> ()>
module {
  func.func @scalar() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<f32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<f32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
    %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<writeonly:tensor<f32>> -> tensor<f32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%2 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<f32>
    flow.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
    return
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: func.func @scalar()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @transpose_8x8() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
    %3 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<512x1024xf32>) outs(%3 : tensor<1024x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1024x512xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [8, 8], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2,+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @transpose_16x16() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
    %3 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<512x1024xf32>) outs(%3 : tensor<1024x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1024x512xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x512xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [16, 16], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @multi_root() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %c6144 = arith.constant 6144 : index
    %c792576 = arith.constant 792576 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<12x128xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c792576) : !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128x128xf32>> -> tensor<12x128x128xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [12, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128xf32>> -> tensor<12x128xf32>
    %5 = tensor.empty() : tensor<12x128xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<12x128xf32>) -> tensor<12x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3 : tensor<12x128x128xf32>) outs(%4 : tensor<12x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.maximumf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<12x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%6 : tensor<12x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.subf %in, %in_0 : f32
      %10 = math.exp %9 : f32
      %11 = arith.addf %10, %out : f32
      linalg.yield %11 : f32
    } -> tensor<12x128xf32>
    flow.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [12, 128], strides = [1, 1] : tensor<12x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 32], [1, 4], [0, 0], [0, 0]]
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 32, 0], [1, 4, 0], [0, 0, 4], [0, 0, 0]]
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @multi_root()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.fill
// CHECK-SAME:     lowering_config = #[[CONFIG1]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG2]]
//      CHECK: linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG2]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<20x40xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [20, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20x40xf32>> -> tensor<20x40xf32>
    %3 = tensor.empty() : tensor<2x48x16x1xf32>
    %pack = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : tensor<20x40xf32> -> tensor<2x48x16x1xf32>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [2, 48, 16, 1], strides = [1, 1, 1, 1] : tensor<2x48x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8], [1, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: func.func @pack()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @pack_f16() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<20x40xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [20, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20x40xf16>> -> tensor<20x40xf16>
    %3 = tensor.empty() : tensor<2x48x16x1xf16>
    %pack = tensor.pack %2 padding_value(%cst : f16) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : tensor<20x40xf16> -> tensor<2x48x16x1xf16>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [2, 48, 16, 1], strides = [1, 1, 1, 1] : tensor<2x48x16x1xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x48x16x1xf16>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8], [1, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: func.func @pack_f16()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @pack_many_elements() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1200x500000xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<31250x1200x16x1xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1200, 500000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1200x500000xf32>> -> tensor<1200x500000xf32>
    %3 = tensor.empty() : tensor<31250x1200x16x1xf32>
    %pack = tensor.pack %2 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %3 : tensor<1200x500000xf32> -> tensor<31250x1200x16x1xf32>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [31250, 1200, 16, 1], strides = [1, 1, 1, 1] : tensor<31250x1200x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<31250x1200x16x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[50, 3], [1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: func.func @pack_many_elements()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @unpack_generic_pack(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
    %5 = tensor.empty() : tensor<24x512x16x1xf32>
    %6 = tensor.empty() : tensor<384x512xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %6 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %unpack : tensor<512xf32>, tensor<384x512xf32>) outs(%6 : tensor<384x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.addf %in, %in_1 : f32
      %9 = arith.minimumf %8, %cst : f32
      %10 = arith.maximumf %9, %cst_0 : f32
      linalg.yield %10 : f32
    } -> tensor<384x512xf32>
    %pack = tensor.pack %7 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %5 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
    flow.dispatch.tensor.store %pack, %2, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 64], [1, 4], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [16, 4], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @unpack_generic_pack(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG1]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elem_pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x384x8x1xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
    %3 = tensor.empty() : tensor<128x384xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x384xf32>) outs(%3 : tensor<128x384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %in : f32
      linalg.yield %6 : f32
    } -> tensor<128x384xf32>
    %5 = tensor.empty() : tensor<16x384x8x1xf32>
    %pack = tensor.pack %4 inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %5 : tensor<128x384xf32> -> tensor<16x384x8x1xf32>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [16, 384, 8, 1], strides = [1, 1, 1, 1] : tensor<16x384x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x384x8x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [8, 1], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 64], [1, 1], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @elem_pack()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf", ukernels = false}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @transpose_pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c1579008 = arith.constant 1579008 : index
    %c3147776 = arith.constant 3147776 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1579008) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<30522x768xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c3147776) : !flow.dispatch.tensor<writeonly:tensor<1908x768x16x1xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [30522, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<30522x768xf32>> -> tensor<30522x768xf32>
    %3 = tensor.empty() : tensor<768x30522xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<30522x768xf32>) outs(%3 : tensor<768x30522xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x30522xf32>
    %5 = tensor.empty() : tensor<1908x768x16x1xf32>
    %pack = tensor.pack %4 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %5 : tensor<768x30522xf32> -> tensor<1908x768x16x1xf32>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [1908, 768, 16, 1], strides = [1, 1, 1, 1] : tensor<1908x768x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1908x768x16x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [1, 16], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 64], [1, 1], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @transpose_pack()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG2]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf", ukernels = false}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @reduction_broadcast_pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant -0.000000e+00 : f32
    %cst_0 = arith.constant 1.024000e+03 : f32
    %cst_1 = arith.constant 9.99999996E-13 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x1024xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024xf32>>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<24x1024x16x1xf32>>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x1024xf32>> -> tensor<384x1024xf32>
    %6 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
    %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1024], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1024xf32>> -> tensor<1024xf32>
    %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [1024], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1024xf32>> -> tensor<1024xf32>
    %9 = tensor.empty() : tensor<24x1024x16x1xf32>
    %10 = tensor.empty() : tensor<384x1024xf32>
    %11 = tensor.empty() : tensor<384xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<384xf32>) -> tensor<384xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%5, %6 : tensor<384x1024xf32>, tensor<384xf32>) outs(%12 : tensor<384xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %15 = arith.subf %in, %in_2 : f32
      %16 = arith.mulf %15, %15 : f32
      %17 = arith.addf %out, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<384xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map2, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %13, %7, %8, %6 : tensor<384x1024xf32>, tensor<384xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<384xf32>) outs(%10 : tensor<384x1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %in_5: f32, %out: f32):
      %15 = arith.divf %in_2, %cst_0 : f32
      %16 = arith.addf %15, %cst_1 : f32
      %17 = math.rsqrt %16 : f32
      %18 = arith.mulf %17, %in_3 : f32
      %19 = arith.mulf %in_5, %18 : f32
      %20 = arith.subf %in_4, %19 : f32
      %21 = arith.mulf %in, %18 : f32
      %22 = arith.addf %21, %20 : f32
      linalg.yield %22 : f32
    } -> tensor<384x1024xf32>
    %pack = tensor.pack %14 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %9 : tensor<384x1024xf32> -> tensor<24x1024x16x1xf32>
    flow.dispatch.tensor.store %pack, %4, offsets = [0, 0, 0, 0], sizes = [24, 1024, 16, 1], strides = [1, 1, 1, 1] : tensor<24x1024x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x1024x16x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32], [16], [0], [0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 0], [16, 0], [0, 16], [0, 0]]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 0], [16, 0], [0, 0], [0, 1]]>
//  CHECK-DAG: #[[CONFIG4:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[2, 0], [1, 0], [0, 0], [0, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @reduction_broadcast_pack()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG4]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf", ukernels = false}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @reduction_pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant -0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x1024x32xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x1024xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x24x16x1xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [384, 1024, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<384x1024x32xf32>> -> tensor<384x1024x32xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x1024xf32>> -> tensor<384x1024xf32>
    %5 = tensor.empty() : tensor<1024x24x16x1xf32>
    %6 = tensor.empty() : tensor<384x1024x32xf32>
    %7 = tensor.empty() : tensor<384x1024xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<384x1024xf32>) -> tensor<384x1024xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<384x1024x32xf32>, tensor<384x1024xf32>) outs(%8 : tensor<384x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %10 = arith.subf %in, %in_0 : f32
      %11 = arith.mulf %10, %10 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<384x1024xf32>
    %pack = tensor.pack %9 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %5 : tensor<384x1024xf32> -> tensor<1024x24x16x1xf32>
    flow.dispatch.tensor.store %pack, %2, offsets = [0, 0, 0, 0], sizes = [1024, 24, 16, 1], strides = [1, 1, 1, 1] : tensor<1024x24x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x24x16x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32], [16, 1], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32, 0], [16, 1, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 2], [1, 1], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @reduction_pack()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.fill
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   tensor.pack
// CHECK-SAME:       lowering_config = #[[CONFIG3]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @unpack_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c41943040 = arith.constant 41943040 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c41943040) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256x16x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x4096xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 256, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256x16x16xf32>> -> tensor<64x256x16x16xf32>
    %3 = tensor.empty() : tensor<1024x4096xf32>
    %unpack = tensor.unpack %2 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<64x256x16x16xf32> -> tensor<1024x4096xf32>
    flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : tensor<1024x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x4096xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [16, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: func.func @unpack_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @unpack_outer_dims_perm_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c41943040 = arith.constant 41943040 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c41943040) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64x256x16x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x4096x64xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [64, 64, 256, 16, 16], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64x256x16x16xf32>> -> tensor<64x64x256x16x16xf32>
    %3 = tensor.empty() : tensor<1024x4096x64xf32>
    %unpack = tensor.unpack %2 outer_dims_perm = [2, 0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<64x64x256x16x16xf32> -> tensor<1024x4096x64xf32>
    flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0, 0], sizes = [1024, 4096, 64], strides = [1, 1, 1] : tensor<1024x4096x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x4096x64xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 64], [16, 16, 16]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDataTiling>
//      CHECK: func.func @unpack_outer_dims_perm_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.unpack
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @unpack_elem() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<48x64x8x2xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [48, 64, 8, 2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<48x64x8x2xf32>> -> tensor<48x64x8x2xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
    %5 = tensor.empty() : tensor<128x384xf32>
    %6 = tensor.empty() : tensor<384x128xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %6 : tensor<48x64x8x2xf32> -> tensor<384x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%4, %unpack : tensor<128xf32>, tensor<384x128xf32>) outs(%5 : tensor<128x384xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<128x384xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64], [2, 8], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @unpack_elem()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @quant_model() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %c12_i32 = arith.constant 12 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2304x24xi8>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<24x144xi8>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<144xi32>>
    %3 = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2304x144xi8>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2304, 24], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2304x24xi8>> -> tensor<2304x24xi8>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [24, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24x144xi8>> -> tensor<24x144xi8>
    %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [144], strides = [1] : !flow.dispatch.tensor<readonly:tensor<144xi32>> -> tensor<144xi32>
    %7 = tensor.empty() : tensor<2304x144xi8>
    %8 = tensor.empty() : tensor<2304x144xi32>
    %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
    %10 = linalg.matmul ins(%4, %5 : tensor<2304x24xi8>, tensor<24x144xi8>) outs(%9 : tensor<2304x144xi32>) -> tensor<2304x144xi32>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %10 : tensor<144xi32>, tensor<2304x144xi32>) outs(%7 : tensor<2304x144xi8>) {
    ^bb0(%in: i32, %in_0: i32, %out: i8):
      %12 = arith.subi %in_0, %c12_i32 : i32
      %13 = arith.addi %in, %12 : i32
      %14 = arith.trunci %13 : i32 to i8
      linalg.yield %14 : i8
    } -> tensor<2304x144xi8>
    flow.dispatch.tensor.store %11, %3, offsets = [0, 0], sizes = [2304, 144], strides = [1, 1] : tensor<2304x144xi8> -> !flow.dispatch.tensor<writeonly:tensor<2304x144xi8>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 48, 0], [64, 48, 0], [0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @quant_model()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = false}>
module {
  func.func @test() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %c6364136223846793005_i64 = arith.constant 6364136223846793005 : i64
    %c1442695040888963407_i64 = arith.constant 1442695040888963407 : i64
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<i64>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<i64>>
    %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %3 = arith.muli %extracted, %c6364136223846793005_i64 : i64
    %4 = arith.addi %3, %c1442695040888963407_i64 : i64
    %inserted = tensor.insert %4 into %2[] : tensor<i64>
    flow.dispatch.tensor.store %inserted, %1, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
    return
  }
}

//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK:   func.func @test()
// CHECK-SAME:       translation_info = #[[TRANSLATION]]

// -----

#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {cpu = "cascadelake", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", link_embedded = false, native_vector_size = 64 : index, target_triple = "x86_64-unknown-linux-gnu", ukernels = false}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @non_trivial_program() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x1x128x1xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x1xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 1, 128, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1x128x1xf32>> -> tensor<128x1x128x1xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1xf32>> -> tensor<128x1xf32>
    %5 = tensor.empty() : tensor<1x1xf32>
    %6 = tensor.empty() : tensor<128xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<128xf32>) -> tensor<128xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1], [2, 3]] : tensor<128x1x128x1xf32> into tensor<128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<128x128xf32>) outs(%7 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %out, %in : f32
      linalg.yield %11 : f32
    } -> tensor<128xf32>
    %expanded = tensor.expand_shape %9 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
    %10 = linalg.matmul ins(%expanded, %4 : tensor<1x128xf32>, tensor<128x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.dispatch.tensor.store %10, %2, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-NOT:   lowering_config
//      CHECK: func.func @non_trivial_program()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = true}>
module {
  func.func @batch_mmt4d() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = hal.interface.constant.load[3] : i32
    %4 = arith.extui %0 : i32 to i64
    %5 = arith.extui %1 : i32 to i64
    %6 = arith.shli %5, %c32_i64 : i64
    %7 = arith.ori %4, %6 : i64
    %8 = arith.index_castui %7 {stream.alignment = 64 : index} : i64 to index
    %9 = arith.extui %2 : i32 to i64
    %10 = arith.extui %3 : i32 to i64
    %11 = arith.shli %10, %c32_i64 : i64
    %12 = arith.ori %9, %11 : i64
    %13 = arith.index_castui %12 : i64 to index
    %14 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x10x32x8x1xf32>>
    %15 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x80x32x4x1xf32>>
    %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%13) : !flow.dispatch.tensor<writeonly:tensor<128x10x80x8x4xf32>>
    %17 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 32, 8, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x10x32x8x1xf32>> -> tensor<128x10x32x8x1xf32>
    %18 = flow.dispatch.tensor.load %15, offsets = [0, 0, 0, 0, 0], sizes = [128, 80, 32, 4, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x80x32x4x1xf32>> -> tensor<128x80x32x4x1xf32>
    %19 = tensor.empty() : tensor<128x10x80x8x4xf32>
    %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
    %21 = linalg.batch_mmt4d ins(%17, %18 : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>) outs(%20 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
    flow.dispatch.tensor.store %21, %16, offsets = [0, 0, 0, 0, 0], sizes = [128, 10, 80, 8, 4], strides = [1, 1, 1, 1, 1] : tensor<128x10x80x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x10x80x8x4xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 10, 80, 0, 0, 0, 0], [1, 10, 80, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 8, 4, 0], [0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]{{\]}}>
//      CHECK: func.func @batch_mmt4d()
//      CHECK:   linalg.batch_mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "cascadelake", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
module {
  func.func @mmt4d_with_large_reduction() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x18176x16x1xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<284x18176x16x1xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<7x284x16x16xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [7, 18176, 16, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x18176x16x1xf32>> -> tensor<7x18176x16x1xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [284, 18176, 16, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<284x18176x16x1xf32>> -> tensor<284x18176x16x1xf32>
    %5 = tensor.empty() : tensor<7x284x16x16xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<7x284x16x16xf32>) -> tensor<7x284x16x16xf32>
    %7 = linalg.mmt4d ins(%3, %4 : tensor<7x18176x16x1xf32>, tensor<284x18176x16x1xf32>) outs(%6 : tensor<7x284x16x16xf32>) -> tensor<7x284x16x16xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [7, 284, 16, 16], strides = [1, 1, 1, 1] : tensor<7x284x16x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<7x284x16x16xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 0, 2, 16, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]>
//      CHECK: func.func @mmt4d_with_large_reduction()
//      CHECK:   linalg.mmt4d
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @pad_only() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c634816 = arith.constant 634816 : index
    %c3846080 = arith.constant 3846080 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c634816) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x112x112x64xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c3846080) : !flow.dispatch.tensor<writeonly:tensor<1x114x114x64xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x112x112x64xf32>> -> tensor<1x112x112x64xf32>
    %padded = tensor.pad %2 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst : f32
    } : tensor<1x112x112x64xf32> to tensor<1x114x114x64xf32>
    flow.dispatch.tensor.store %padded, %1, offsets = [0, 0, 0, 0], sizes = [1, 114, 114, 64], strides = [1, 1, 1, 1] : tensor<1x114x114x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x114x114x64xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 6, 57, 64], [1, 1, 1, 4], [0, 0, 0, 0], [0, 0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @pad_only()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   tensor.pad {{.+}} {
//      CHECK:     tensor.yield
// CHECK-NEXT:   } {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
    func.func @winograd_output_transform() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>> -> tensor<8x8x2x6x6x128xf16>
    %3 = tensor.empty() : tensor<2x36x36x128xf16>
    %4 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<8x8x2x6x6x128xf16>) outs(%3 : tensor<2x36x36x128xf16>) -> tensor<2x36x36x128xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 36, 36, 128], strides = [1, 1, 1, 1] : tensor<2x36x36x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 6, 64], [1, 1, 1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPULinalgExtTileAndVectorize>
//      CHECK: func.func @winograd_output_transform()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.winograd.output_transform
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
    func.func @winograd_input_transform() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>> -> tensor<2x34x34x128xf16>
    %3 = tensor.empty() : tensor<8x8x2x6x6x128xf16>
    %4 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<8x8x2x6x6x128xf16>) -> tensor<8x8x2x6x6x128xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x2x6x6x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 6, 64], [1, 1, 1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPULinalgExtTileAndVectorize>
//      CHECK: func.func @winograd_input_transform()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.winograd.input_transform
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
    func.func @winograd_filter_transform() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8x64x128xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>> -> tensor<3x3x64x128xf32>
    %3 = tensor.empty() : tensor<8x8x64x128xf32>
    %4 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1]) ins(%2 : tensor<3x3x64x128xf32>) outs(%3 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [8, 8, 64, 128], strides = [1, 1, 1, 1] : tensor<8x8x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x8x64x128xf32>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 64], [1, 1]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPULinalgExtTileAndVectorize>
//      CHECK: func.func @winograd_filter_transform()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.winograd.filter_transform
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @attention() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %scale = arith.constant 0.125 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
    %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
    %7 = tensor.empty() : tensor<20x4096x64xf16>
    %8 = iree_linalg_ext.attention
      ins(%4, %5, %6, %scale : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16)
      outs(%7 : tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
    flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[20, 64, 0, 0, 0], [20, 32, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPULinalgExtTileAndVectorize>
//      CHECK: func.func @attention()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//     CHECK:   iree_linalg_ext.attention
// CHECK-SAME:    {lowering_config = #[[CONFIG]]}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
      cpu = "generic", cpu_features = "",
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @elementwise_output_transposed() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<i64>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<768xi64>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32xi64>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x32x768xf32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
    %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [768], strides = [1] : !flow.dispatch.tensor<readonly:tensor<768xi64>> -> tensor<768xi64>
    %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xi64>> -> tensor<32xi64>
    %7 = tensor.empty() : tensor<32x32x768xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5, %6 : tensor<i64>, tensor<768xi64>, tensor<32xi64>) outs(%7 : tensor<32x32x768xf32>) {
    ^bb0(%in: i64, %in_0: i64, %in_1: i64, %out: f32):
      %9 = arith.addi %in, %in_0 : i64
      %10 = arith.addi %9, %in_1 : i64
      %11 = arith.uitofp %10 : i64 to f32
      linalg.yield %11 : f32
    } -> tensor<32x32x768xf32>
    flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [32, 32, 768], strides = [1, 1, 1] : tensor<32x32x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x32x768xf32>>
    return
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 32, 32], [1, 8, 1], [0, 0, 0], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @elementwise_output_transposed()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//     CHECK:    linalg.generic
// CHECK-SAME:     {lowering_config = #[[CONFIG]]}
