// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

// Check that this dispatch compiles to vectors and that there are no allocas.
// By proxy checks that destination passing style kicked in correctly
// and no CSE was run between first level tile + fuse + distribute
// and the conversion to destination passing style. Running CSE
// before hoists the fill and the empty out of the loop causing
// issues with the conversion.
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
func.func @check_no_cse() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 3.840000e+02 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_cast %0 {stream.alignment = 512 : index, stream.values = [0 : index, 10752 : index]} : i32 to index
  %3 = arith.index_cast %1 {stream.alignment = 512 : index, stream.values = [10752 : index, 21504 : index]} : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%2) : !flow.dispatch.tensor<readonly:tensor<7x384xf32>>
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<7xf32>>
  %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [7, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x384xf32>> -> tensor<7x384xf32>
  %7 = tensor.empty() : tensor<7xf32>
  %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<7xf32>) -> tensor<7xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<7x384xf32>) outs(%8 : tensor<7xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = arith.addf %out, %in : f32
    linalg.yield %11 : f32
  } -> tensor<7xf32>
  %10 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%9 : tensor<7xf32>) outs(%7 : tensor<7xf32>) {
  ^bb0(%in: f32, %out: f32):
    %11 = arith.divf %in, %cst : f32
    linalg.yield %11 : f32
  } -> tensor<7xf32>
  flow.dispatch.tensor.store %10, %5, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:tensor<7xf32>>
  return
}
// CHECK-LABEL: func.func @check_no_cse()
//   CHECK-NOT:    memref.alloc
//       CHECK:    scf.for
//       CHECK:      arith.addf
//       CHECK:    vector.reduction <add>
//       CHECK:    arith.divf
//       CHECK:    memref.store

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @peel_partially_unaligned_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = arith.index_castui %0 {stream.alignment = 128 : index, stream.values = [0 : index, 131712 : index]} : i32 to index
  %5 = arith.index_castui %1 {stream.alignment = 64 : index, stream.values = [576704 : index, 1763072 : index]} : i32 to index
  %6 = arith.index_castui %2 {stream.alignment = 64 : index, stream.values = [908480 : index, 2094848 : index]} : i32 to index
  %7 = arith.index_castui %3 {stream.alignment = 128 : index, stream.values = [2304 : index, 134016 : index]} : i32 to index
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x576xf32>>
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<576x144xf32>>
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x144xf32>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<1x144xf32>>
  %12 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [1, 576], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x576xf32>> -> tensor<1x576xf32>
  %13 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [576, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<576x144xf32>> -> tensor<576x144xf32>
  %14 = flow.dispatch.tensor.load %10, offsets = [0, 0], sizes = [1, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x144xf32>> -> tensor<1x144xf32>
  %15 = tensor.empty() : tensor<1x144xf32>
  %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1x144xf32>) -> tensor<1x144xf32>
  %17 = linalg.matmul ins(%12, %13 : tensor<1x576xf32>, tensor<576x144xf32>) outs(%16 : tensor<1x144xf32>) -> tensor<1x144xf32>
  %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %14 : tensor<1x144xf32>, tensor<1x144xf32>) outs(%15 : tensor<1x144xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.addf %in, %in_0 : f32
    %20 = arith.maximumf %19, %cst : f32
    linalg.yield %20 : f32
  } -> tensor<1x144xf32>
  flow.dispatch.tensor.store %18, %11, offsets = [0, 0], sizes = [1, 144], strides = [1, 1] : tensor<1x144xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x144xf32>>
  return
}
// Checks that the bounded stack allocation are created.
// CHECK-LABEL: func.func @peel_partially_unaligned_matmul
// Main loop:
//       CHECK:     vector.fma
//       CHECK:     arith.addf {{.*}} : vector<
//       CHECK:     arith.maximumf {{.*}} : vector<
//
// Peeled loop:
//       CHECK:     vector.fma
//       CHECK:     arith.addf {{.*}} : vector<
//       CHECK:     arith.maximumf {{.*}} : vector<

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @batch_matmul_dynamic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
  %6 = arith.index_cast %0 : i32 to index
  %7 = arith.index_cast %1 : i32 to index
  %8 = arith.index_cast %2 : i32 to index
  %9 = arith.index_cast %3 : i32 to index
  %10 = arith.index_cast %4 : i32 to index
  %11 = arith.index_cast %5 : i32 to index
  %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%6, %7, %9}
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%10, %11, %8}
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%6, %7, %8}
  %15 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0], sizes = [%6, %7, %9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%6, %7, %9} -> tensor<?x?x?xf32>
  %16 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [%10, %11, %8], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%10, %11, %8} -> tensor<?x?x?xf32>
  %17 = tensor.empty(%6, %7, %8) : tensor<?x?x?xf32>
  %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %19 = linalg.batch_matmul ins(%15, %16 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  flow.dispatch.tensor.store %19, %14, offsets = [0, 0, 0], sizes = [%6, %7, %8], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%6, %7, %8}
  return
}
// CHECK-LABEL: func.func @batch_matmul_dynamic
//       CHECK:   vector.fma

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0 * 1536 + d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @check_buffer_ops_vectorization() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<128x1024xi32>
  memref.assume_alignment %0, 64 : memref<128x1024xi32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<128x1536xi32>
  memref.assume_alignment %1, 64 : memref<128x1536xi32>
  %subview = memref.subview %1[0, 0] [128, 1024] [1, 1] : memref<128x1536xi32> to memref<128x1024xi32, #map>
  linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<128x1024xi32>) outs(%subview : memref<128x1024xi32, #map>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}
// CHECK-LABEL:  #{{.+}} = #iree_codegen.translation_info<pipeline = CPUBufferOpsTileAndVectorize
//       CHECK:      func.func @check_buffer_ops_vectorization
//       CHECK:        vector.load
//       CHECK:        vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @vectorize_fill_conv2d_generic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 3.000000e+00 : f32
  %cst_1 = arith.constant 6.000000e+00 : f32
  %cst_2 = arith.constant 0.166666672 : f32
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
  %5 = tensor.empty() : tensor<1x112x112x16xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3, %7 : tensor<16xf32>, tensor<1x112x112x16xf32>) outs(%5 : tensor<1x112x112x16xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %9 = arith.addf %in, %in_4 : f32
    %10 = arith.addf %9, %cst_0 : f32
    %11 = arith.cmpf olt, %10, %cst : f32
    %12 = arith.select %11, %cst, %10 : f32
    %13 = arith.cmpf olt, %cst_1, %10 : f32
    %14 = arith.select %13, %cst_1, %12 : f32
    %15 = arith.mulf %9, %14 : f32
    %16 = arith.mulf %15, %cst_2 : f32
    linalg.yield %16 : f32
  } -> tensor<1x112x112x16xf32>
  flow.dispatch.tensor.store %8, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
  return
}

// CHECK-LABEL:  func.func @vectorize_fill_conv2d_generic
//   CHECK-NOT:    memref.alloca
//   CHECK-NOT:    linalg.fill
//       CHECK:    vector.fma
//   CHECK-NOT:    linalg.generic
//       CHECK:    arith.cmpf olt, %{{.+}}, %{{.+}} : vector<4x4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
func.func @multi_result() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e-03 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<writeonly:tensor<128x256xf32>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(5) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
  %6 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %8 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
  %9 = tensor.empty() : tensor<64x256xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %11 = linalg.matmul ins(%6, %7 : tensor<64x128xf32>, tensor<128x256xf32>) outs(%10 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %8 : tensor<64x256xf32>, tensor<256xf32>) outs(%9 : tensor<64x256xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %13 = arith.addf %in, %in_1 : f32
    linalg.yield %13 : f32
  } -> tensor<64x256xf32>
  flow.dispatch.tensor.store %11, %4, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
  flow.dispatch.tensor.store %12, %5, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256xf32>>
  return
}
//    CHECK-LABEL: func @multi_result
//          CHECK:   scf.for
//          CHECK:     scf.for
//          CHECK:       scf.for
// CHECK-COUNT-16:         vector.fma
//          CHECK:       arith.addf %{{.+}}, %{{.+}} : vector<8x32xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf", ukernels = "mmt4d"}>
func.func @ukernel_dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4x8x32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x4x16x32xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 4, 8, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x8x32xf32>> -> tensor<2x4x8x32xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [16, 4, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4x16x32xf32>> -> tensor<16x4x16x32xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>> -> tensor<2x16x8x16xf32>
  %6 = linalg.mmt4d ins(%3, %4 : tensor<2x4x8x32xf32>, tensor<16x4x16x32xf32>) outs(%5 : tensor<2x16x8x16xf32>) -> tensor<2x16x8x16xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 16], strides = [1, 1, 1, 1] : tensor<2x16x8x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>>
  return
}
// CHECK-LABEL: func @ukernel_dispatch()
// Checks scf.for for distribution loops.
//       CHECK:   scf.forall
// Checks scf.for for outer and inner parallel loops.
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:           scf.for
//   CHECK-NOT:             scf.for
//       CHECK:   iree_codegen.ukernel.generic "iree_uk_mmt4d"

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf", ukernels = "all"}>
#map = affine_map<()[s0, s1, s2] -> (s0 - s1 * (s0 ceildiv s2), s0 ceildiv s2)>
#map1 = affine_map<()[s0, s1, s2] -> (s0 * (s1 ceildiv s2))>
func.func @dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_castui %0 : i32 to index
  %3 = arith.index_castui %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?xf32>>{%2}
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %7 = affine.min #map()[%2, %workgroup_id_x, %workgroup_count_x]
  %8 = affine.apply #map1()[%workgroup_id_x, %2, %workgroup_count_x]
  %9 = flow.dispatch.tensor.load %4, offsets = [%8], sizes = [%7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%2} -> tensor<?xf32>
  %10 = flow.dispatch.tensor.load %5, offsets = [%8], sizes = [%7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%3} -> tensor<?xf32>
  %11 = tensor.empty(%7) : tensor<?xf32>
  %12 = iree_codegen.ukernel.generic "simple_mul_workgroup" ins(%9, %10 : tensor<?xf32>, tensor<?xf32>) outs(%11 : tensor<?xf32>) (%7 : index) -> tensor<?xf32>
  flow.dispatch.tensor.store %12, %6, offsets = [%8], sizes = [%7], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?xf32>>{%2}
  return
}
//       CHECK:   func @dispatch
//       CHECK:     %[[INPUT0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[INPUT1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[OUTPUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:     %[[OFFSET:.+]] = affine.apply
//   CHECK-DAG:     %[[SIZE:.+]] = affine.min
//   CHECK-DAG:     %[[SUBVIEW_OUTPUT:.+]] = memref.subview %[[OUTPUT]][%[[OFFSET]]] [%[[SIZE]]]
//   CHECK-DAG:     %[[SUBVIEW_INPUT0:.+]] = memref.subview %[[INPUT0]][%[[OFFSET]]] [%[[SIZE]]]
//   CHECK-DAG:     %[[SUBVIEW_INPUT1:.+]] = memref.subview %[[INPUT1]][%[[OFFSET]]] [%[[SIZE]]]
//       CHECK:     iree_codegen.ukernel.generic "simple_mul_workgroup"
//  CHECK-SAME:         ins(%[[SUBVIEW_INPUT0]], %[[SUBVIEW_INPUT1]]
//  CHECK-SAME:         outs(%[[SUBVIEW_OUTPUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 2, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]>
#config1 = #iree_codegen.lowering_config<tile_sizes = [[1, 2, 0, 0, 0, 0], [1, 1, 0, 1, 128, 0], [0, 0, 1, 0, 0, 1]]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+fma,+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = "all"}>
func.func @unsupported_ukernel_fallback_to_vectorization() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_, translation_info = #iree_codegen.translation_info<pipeline = Mmt4dTilingExpert>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c132096 = arith.constant 132096 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x256x1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1024) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x256x128x1xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c132096) : !flow.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 256, 1, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256x1x1xf32>> -> tensor<1x256x1x1xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 256, 128, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x256x128x1xi8>> -> tensor<4x256x128x1xi8>
  %5 = tensor.empty() : tensor<1x4x1x128xf32>
  %6 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%5 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  %7 = linalg.mmt4d {lowering_config = #config1} ins(%3, %4 : tensor<1x256x1x1xf32>, tensor<4x256x128x1xi8>) outs(%6 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 1, 128], strides = [1, 1, 1, 1] : tensor<1x4x1x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
  return
}
// CHECK-LABEL: func.func @unsupported_ukernel_fallback_to_vectorization
// CHECK:         vector.fma

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+fma,+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @dequant_matmul() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c132096 = arith.constant 132096 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x256x1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1024) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x256x128x1xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c132096) : !flow.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 256, 1, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256x1x1xf32>> -> tensor<1x256x1x1xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 256, 128, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x256x128x1xi8>> -> tensor<4x256x128x1xi8>
  %5 = tensor.empty() : tensor<1x4x1x128xf32>
  %8 = tensor.empty() : tensor<4x256x128x1xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<4x256x128x1xi8>) outs(%8 : tensor<4x256x128x1xf32>) {
    ^bb0(%in: i8, %out: f32):
      %10 = arith.uitofp %in : i8 to f32
      linalg.yield %10 : f32
    } -> tensor<4x256x128x1xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  %7 = linalg.mmt4d  ins(%3, %9 : tensor<1x256x1x1xf32>, tensor<4x256x128x1xf32>) outs(%6 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 1, 128], strides = [1, 1, 1, 1] : tensor<1x4x1x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
  return
}
// CHECK-LABEL: func.func @dequant_matmul
// Checks whether the dequant and fma operation are fused inside the reduction loop.
// CHECK:      scf.for
// CHECK:        arith.uitofp
// CHECK:        vector.fma

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+fma,+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = "all"}>
func.func @fuse_inputs_reduction() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<64x1x1x16x16xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<64x16x16xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [64, 1, 1, 16, 16], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1x1x16x16xf32>> -> tensor<64x1x1x16x16xf32>
  %3 = tensor.empty() : tensor<64x16x16xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x16x16xf32>) -> tensor<64x16x16xf32>
  %unpack = linalg.unpack %2 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [16, 16] into %3 : tensor<64x1x1x16x16xf32> -> tensor<64x16x16xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%unpack : tensor<64x16x16xf32>) outs(%4 : tensor<64x16x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %out, %in : f32
    linalg.yield %6 : f32
  } -> tensor<64x16x16xf32>
  flow.dispatch.tensor.store %5, %1, offsets = [0, 0, 0], sizes = [64, 16, 16], strides = [1, 1, 1] : tensor<64x16x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x16x16xf32>>
  return
}
// CHECK-LABEL: func.func @fuse_inputs_reduction
//     CHECK:      scf.for
//     CHECK:        vector.load
// CHECK-NOT:        scf.for
//     CHECK:        arith.addf
