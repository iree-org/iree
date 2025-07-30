// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-llvmcpu-configuration-pipeline, func.func(iree-llvmcpu-lower-executable-target, iree-llvmcpu-check-ir-before-llvm-conversion))' --split-input-file %s | FileCheck %s

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
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%2) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x384xf32>>
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%3) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7xf32>>
  %6 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [7, 384], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x384xf32>> -> tensor<7x384xf32>
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
  iree_tensor_ext.dispatch.tensor.store %10, %5, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7xf32>>
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
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x576xf32>>
  %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%5) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x144xf32>>
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%6) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x144xf32>>
  %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%7) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x144xf32>>
  %12 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [1, 576], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x576xf32>> -> tensor<1x576xf32>
  %13 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [0, 0], sizes = [576, 144], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x144xf32>> -> tensor<576x144xf32>
  %14 = iree_tensor_ext.dispatch.tensor.load %10, offsets = [0, 0], sizes = [1, 144], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x144xf32>> -> tensor<1x144xf32>
  %15 = tensor.empty() : tensor<1x144xf32>
  %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1x144xf32>) -> tensor<1x144xf32>
  %17 = linalg.matmul ins(%12, %13 : tensor<1x576xf32>, tensor<576x144xf32>) outs(%16 : tensor<1x144xf32>) -> tensor<1x144xf32>
  %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %14 : tensor<1x144xf32>, tensor<1x144xf32>) outs(%15 : tensor<1x144xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.addf %in, %in_0 : f32
    %20 = arith.maximumf %19, %cst : f32
    linalg.yield %20 : f32
  } -> tensor<1x144xf32>
  iree_tensor_ext.dispatch.tensor.store %18, %11, offsets = [0, 0], sizes = [1, 144], strides = [1, 1] : tensor<1x144xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x144xf32>>
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
  %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%6, %7, %9}
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%10, %11, %8}
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%6, %7, %8}
  %15 = iree_tensor_ext.dispatch.tensor.load %12, offsets = [0, 0, 0], sizes = [%6, %7, %9], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%6, %7, %9} -> tensor<?x?x?xf32>
  %16 = iree_tensor_ext.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [%10, %11, %8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%10, %11, %8} -> tensor<?x?x?xf32>
  %17 = tensor.empty(%6, %7, %8) : tensor<?x?x?xf32>
  %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %19 = linalg.batch_matmul ins(%15, %16 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  iree_tensor_ext.dispatch.tensor.store %19, %14, offsets = [0, 0, 0], sizes = [%6, %7, %8], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%6, %7, %8}
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
  %assume_align_0 = memref.assume_alignment %0, 64 : memref<128x1024xi32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<128x1536xi32>
  %assume_align_1 = memref.assume_alignment %1, 64 : memref<128x1536xi32>
  %subview = memref.subview %assume_align_1[0, 0] [128, 1024] [1, 1] : memref<128x1536xi32> to memref<128x1024xi32, #map>
  linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%assume_align_0 : memref<128x1024xi32>) outs(%subview : memref<128x1024xi32, #map>) {
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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
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
  iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 16], strides = [1, 1, 1, 1] : tensor<1x112x112x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x256xf32>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(5) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>
  %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
  %7 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %8 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
  %9 = tensor.empty() : tensor<64x256xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %11 = linalg.matmul ins(%6, %7 : tensor<64x128xf32>, tensor<128x256xf32>) outs(%10 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %8 : tensor<64x256xf32>, tensor<256xf32>) outs(%9 : tensor<64x256xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %13 = arith.addf %in, %in_1 : f32
    linalg.yield %13 : f32
  } -> tensor<64x256xf32>
  iree_tensor_ext.dispatch.tensor.store %11, %4, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>
  iree_tensor_ext.dispatch.tensor.store %12, %5, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : tensor<64x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x256xf32>>
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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x8x32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4x16x32xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 4, 8, 32], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x8x32xf32>> -> tensor<2x4x8x32xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [16, 4, 16, 32], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4x16x32xf32>> -> tensor<16x4x16x32xf32>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>> -> tensor<2x16x8x16xf32>
  %6 = linalg.mmt4d ins(%3, %4 : tensor<2x4x8x32xf32>, tensor<16x4x16x32xf32>) outs(%5 : tensor<2x16x8x16xf32>) -> tensor<2x16x8x16xf32>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [2, 16, 8, 16], strides = [1, 1, 1, 1] : tensor<2x16x8x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x16x8x16xf32>>
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
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%2}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%3}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf32>>{%2}
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %7 = affine.min #map()[%2, %workgroup_id_x, %workgroup_count_x]
  %8 = affine.apply #map1()[%workgroup_id_x, %2, %workgroup_count_x]
  %9 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [%8], sizes = [%7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%2} -> tensor<?xf32>
  %10 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [%8], sizes = [%7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%3} -> tensor<?xf32>
  %11 = tensor.empty(%7) : tensor<?xf32>
  %12 = iree_codegen.ukernel.generic "simple_mul_workgroup" ins(%9, %10 : tensor<?xf32>, tensor<?xf32>) outs(%11 : tensor<?xf32>) (%7 : index) -> tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %12, %6, offsets = [%8], sizes = [%7], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?xf32>>{%2}
  return
}
//       CHECK:   func @dispatch
//       CHECK:     %[[INPUT0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[ASSUMED_INPUT0:.+]] = memref.assume_alignment %[[INPUT0]], 64
//       CHECK:     %[[INPUT1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[ASSUMED_INPUT1:.+]] = memref.assume_alignment %[[INPUT1]], 64
//       CHECK:     %[[OUTPUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//  CHECK-SAME:         memref<?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:     %[[ASSUMED_OUTPUT:.+]] = memref.assume_alignment %[[OUTPUT]], 64
//   CHECK-DAG:     %[[OFFSET:.+]] = affine.apply
//   CHECK-DAG:     %[[SIZE:.+]] = affine.min
//   CHECK-DAG:     %[[SUBVIEW_OUTPUT:.+]] = memref.subview %[[ASSUMED_OUTPUT]][%[[OFFSET]]] [%[[SIZE]]]
//   CHECK-DAG:     %[[SUBVIEW_INPUT0:.+]] = memref.subview %[[ASSUMED_INPUT0]][%[[OFFSET]]] [%[[SIZE]]]
//   CHECK-DAG:     %[[SUBVIEW_INPUT1:.+]] = memref.subview %[[ASSUMED_INPUT1]][%[[OFFSET]]] [%[[SIZE]]]
//       CHECK:     iree_codegen.ukernel.generic "simple_mul_workgroup"
//  CHECK-SAME:         ins(%[[SUBVIEW_INPUT0]], %[[SUBVIEW_INPUT1]]
//  CHECK-SAME:         outs(%[[SUBVIEW_OUTPUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+fma,+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = "all"}>
func.func @unsupported_ukernel_fallback_to_vectorization() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c132096 = arith.constant 132096 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256x1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1024) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x256x128x1xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c132096) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 256, 1, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256x1x1xf32>> -> tensor<1x256x1x1xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 256, 128, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x256x128x1xi8>> -> tensor<4x256x128x1xi8>
  %5 = tensor.empty() : tensor<1x4x1x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  %7 = linalg.mmt4d ins(%3, %4 : tensor<1x256x1x1xf32>, tensor<4x256x128x1xi8>) outs(%6 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 1, 128], strides = [1, 1, 1, 1] : tensor<1x4x1x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256x1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c1024) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x256x128x1xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c132096) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 256, 1, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256x1x1xf32>> -> tensor<1x256x1x1xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 256, 128, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x256x128x1xi8>> -> tensor<4x256x128x1xi8>
  %5 = tensor.empty() : tensor<1x4x1x128xf32>
  %8 = tensor.empty() : tensor<4x256x128x1xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<4x256x128x1xi8>) outs(%8 : tensor<4x256x128x1xf32>) {
    ^bb0(%in: i8, %out: f32):
      %10 = arith.uitofp %in : i8 to f32
      linalg.yield %10 : f32
    } -> tensor<4x256x128x1xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  %7 = linalg.mmt4d  ins(%3, %9 : tensor<1x256x1x1xf32>, tensor<4x256x128x1xf32>) outs(%6 : tensor<1x4x1x128xf32>) -> tensor<1x4x1x128xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 4, 1, 128], strides = [1, 1, 1, 1] : tensor<1x4x1x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x1x128xf32>>
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
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x1x1x16x16xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x16x16xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [64, 1, 1, 16, 16], strides = [1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x1x1x16x16xf32>> -> tensor<64x1x1x16x16xf32>
  %3 = tensor.empty() : tensor<64x16x16xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x16x16xf32>) -> tensor<64x16x16xf32>
  %unpack = linalg.unpack %2 outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [16, 16] into %3 : tensor<64x1x1x16x16xf32> -> tensor<64x16x16xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%unpack : tensor<64x16x16xf32>) outs(%4 : tensor<64x16x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %out, %in : f32
    linalg.yield %6 : f32
  } -> tensor<64x16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0, 0], sizes = [64, 16, 16], strides = [1, 1, 1] : tensor<64x16x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x16x16xf32>>
  return
}
// CHECK-LABEL: func.func @fuse_inputs_reduction
//     CHECK:      scf.for
//     CHECK:        vector.load
// CHECK-NOT:        scf.for
//     CHECK:        arith.addf

// -----

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = "none"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @mmt4d_bias_relu() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64} {
    %c0 = arith.constant 0 : index
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
    %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
    %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
    %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
    %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
    %5 = arith.index_castui %0 : i32 to index
    %6 = arith.index_castui %1 : i32 to index
    %7 = arith.index_castui %2 : i32 to index
    %8 = arith.index_castui %3 : i32 to index
    %9 = arith.index_castui %4 : i32 to index
    %36 = iree_tensor_ext.dispatch.workload.ordinal %5, 0 : index
    %37 = iree_tensor_ext.dispatch.workload.ordinal %6, 1 : index
    %38 = iree_tensor_ext.dispatch.workload.ordinal %7, 2 : index
    %39 = iree_tensor_ext.dispatch.workload.ordinal %8, 3 : index
    %40 = iree_tensor_ext.dispatch.workload.ordinal %9, 4 : index
    %41 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%39, %36}
    %42 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%37, %40}
    %43 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xf32>>{%38}
    %44 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x16x16xf32>>{%39, %40}
    %45 = iree_tensor_ext.dispatch.tensor.load %41, offsets = [0, 0, 0, 0], sizes = [%39, %36, 16, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%39, %36} -> tensor<?x?x16x1xf32>
    %46 = iree_tensor_ext.dispatch.tensor.load %42, offsets = [0, 0, 0, 0], sizes = [%37, %40, 16, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x16x1xf32>>{%37, %40} -> tensor<?x?x16x1xf32>
    %47 = iree_tensor_ext.dispatch.tensor.load %43, offsets = [0, 0], sizes = [%38, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xf32>>{%38} -> tensor<?x16xf32>
    %48 = tensor.empty(%39, %40) : tensor<?x?x16x16xf32>
    %49 = linalg.fill ins(%cst : f32) outs(%48 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
    %50 = linalg.mmt4d ins(%45, %46 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>) outs(%49 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
    %51 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%50, %47 : tensor<?x?x16x16xf32>, tensor<?x16xf32>) outs(%48 : tensor<?x?x16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %52 = arith.addf %in, %in_0 : f32
      %53 = arith.maximumf %52, %cst : f32
      linalg.yield %53 : f32
    } -> tensor<?x?x16x16xf32>
    iree_tensor_ext.dispatch.tensor.store %51, %44, offsets = [0, 0, 0, 0], sizes = [%39, %40, 16, 16], strides = [1, 1, 1, 1] : tensor<?x?x16x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x16x16xf32>>{%39, %40}
    return
  }
}
// CHECK-LABEL: func.func @mmt4d_bias_relu
// CHECK-NOT:     memref.alloc
// CHECK:         scf.forall
// CHECK:           scf.for
// CHECK:             vector.fma
// CHECK:             vector.insert
// CHECK:           arith.addf

// -----

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", native_vector_size = 64 : i64, ukernels = "none"}>
func.func @mmt4d_unpack_elementwise() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8:4 = util.assume.int
      %4<umin = 127552, umax = 480448>,
      %5<umin = 32768, umax = 98304>,
      %6[<umin = 557056, umax = 557056, udiv = 557056>, <umin = 994816, umax = 994816, udiv = 994816>, <umin = 1432576, umax = 1432576, udiv = 1432576>],
      %7<umin = 32, umax = 96, udiv = 32>
    : index, index, index, index
  %9 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%8#2) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x23x16x1xf16>>
  %10 = iree_tensor_ext.dispatch.workload.ordinal %8#3, 0 : index
  %11 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%8#0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x23x1x1xf16>>{%10}
  %12 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%8#1) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x256xf16>>{%10}
  %13 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x256xf16>>{%10}
  %14 = iree_tensor_ext.dispatch.tensor.load %11, offsets = [0, 0, 0, 0], sizes = [%10, 23, 1, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x23x1x1xf16>>{%10} -> tensor<?x23x1x1xf16>
  %15 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [16, 23, 16, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x23x16x1xf16>> -> tensor<16x23x16x1xf16>
  %16 = iree_tensor_ext.dispatch.tensor.load %12, offsets = [0, 0], sizes = [%10, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x256xf16>>{%10} -> tensor<?x256xf16>
  %17 = tensor.empty(%10) : tensor<?x256xf16>
  %18 = tensor.empty(%10) : tensor<?x16x1x16xf32>
  %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x16x1x16xf32>) -> tensor<?x16x1x16xf32>
  %20 = linalg.mmt4d ins(%14, %15 : tensor<?x23x1x1xf16>, tensor<16x23x16x1xf16>) outs(%19 : tensor<?x16x1x16xf32>) -> tensor<?x16x1x16xf32>
  %21 = tensor.empty(%10) : tensor<?x256xf32>
  %unpack = linalg.unpack %20 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 16] into %21 : tensor<?x16x1x16xf32> -> tensor<?x256xf32>
  %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %unpack : tensor<?x256xf16>, tensor<?x256xf32>) outs(%17 : tensor<?x256xf16>) {
  ^bb0(%in: f16, %in_0: f32, %out: f16):
    %23 = arith.truncf %in_0 : f32 to f16
    %24 = arith.addf %in, %23 : f16
    linalg.yield %24 : f16
  } -> tensor<?x256xf16>
  iree_tensor_ext.dispatch.tensor.store %22, %13, offsets = [0, 0], sizes = [%10, 256], strides = [1, 1] : tensor<?x256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x256xf16>>{%10}
  return
}
// Checks that the stack allocation is in bounds, which implies that the fusion
// happens. Otherwise, it requires a large buffer for intermediate data.
// CHECK-LABEL: func.func @mmt4d_unpack_elementwise

// -----

// This tests that the pack op is not fusible in distribution, and it falls back
// to the iree_linalg_ext.map_scatter solution. The check ensures that the
// iree_linalg_ext.map_scatter op is fused within scf.forall op.

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @pooling_nchw_max_pack_without_padding_issue_20723() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x64x1x256xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x1x1x8x1xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 64, 1, 256], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x64x1x256xf32>> -> tensor<8x64x1x256xf32>
  %3 = tensor.empty() : tensor<1x64x1x1x8x1xf32>
  %4 = tensor.empty() : tensor<8x64x1x1xf32>
  %5 = tensor.empty() : tensor<1x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%4 : tensor<8x64x1x1xf32>) -> tensor<8x64x1x1xf32>
  %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 256]> : vector<2xi64>} ins(%2, %5 : tensor<8x64x1x256xf32>, tensor<1x256xf32>) outs(%6 : tensor<8x64x1x1xf32>) -> tensor<8x64x1x1xf32>
  %pack = linalg.pack %7 outer_dims_perm = [0, 1, 2, 3] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<8x64x1x1xf32> -> tensor<1x64x1x1x8x1xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 64, 1, 1, 8, 1], strides = [1, 1, 1, 1, 1, 1] : tensor<1x64x1x1x8x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x1x1x8x1xf32>>
  return
}
// CHECK-LABEL: func.func @pooling_nchw_max_pack_without_padding_issue_20723(
// CHECK:         scf.forall
// CHECK:           iree_linalg_ext.map_scatter
// CHECK:         } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

// -----

// Similar to the above test, but the pack has padding semantics. After folding
// the padding into iree_linalg_ext.map_scatter op, it creates an additional
// scf.forall to fill the padding values.

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
func.func @pooling_nchw_max_pack_with_padding_issue_20723() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x64x1x256xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x1x1x8x1xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 64, 1, 256], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x64x1x256xf32>> -> tensor<7x64x1x256xf32>
  %3 = tensor.empty() : tensor<1x64x1x1x8x1xf32>
  %4 = tensor.empty() : tensor<7x64x1x1xf32>
  %5 = tensor.empty() : tensor<1x256xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%4 : tensor<7x64x1x1xf32>) -> tensor<7x64x1x1xf32>
  %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<[1, 256]> : vector<2xi64>} ins(%2, %5 : tensor<7x64x1x256xf32>, tensor<1x256xf32>) outs(%6 : tensor<7x64x1x1xf32>) -> tensor<7x64x1x1xf32>
  %cst_0 = arith.constant 0.0 : f32
  %pack = linalg.pack %7 padding_value(%cst_0 : f32) outer_dims_perm = [0, 1, 2, 3] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<7x64x1x1xf32> -> tensor<1x64x1x1x8x1xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 64, 1, 1, 8, 1], strides = [1, 1, 1, 1, 1, 1] : tensor<1x64x1x1x8x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x1x1x8x1xf32>>
  return
}
// CHECK-LABEL: func.func @pooling_nchw_max_pack_with_padding_issue_20723(
// CHECK:         scf.forall
// CHECK:           iree_linalg_ext.map_scatter
// CHECK:         } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
// CHECK:         scf.forall

// -----

// Verify that the dispatch can be compiled without creating large vectors.

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", cpu_features = "", max_stack_allocation_size = 32768 : i64, native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @softmax_dynamic_with_assume_int_hints() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFFC00000 : f32
  %c1 = arith.constant 1 : index
  %c32_i64 = arith.constant 32 : i64
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : i32
  %6 = arith.extui %0 : i32 to i64
  %7 = arith.extui %1 : i32 to i64
  %8 = arith.shli %7, %c32_i64 : i64
  %9 = arith.ori %6, %8 : i64
  %10 = arith.index_castui %9 : i64 to index
  %11 = arith.extui %2 : i32 to i64
  %12 = arith.extui %3 : i32 to i64
  %13 = arith.shli %12, %c32_i64 : i64
  %14 = arith.ori %11, %13 : i64
  %15 = arith.index_castui %14 : i64 to index
  %16 = arith.extui %4 : i32 to i64
  %17 = arith.extui %5 : i32 to i64
  %18 = arith.shli %17, %c32_i64 : i64
  %19 = arith.ori %16, %18 : i64
  %20 = arith.index_castui %19 : i64 to index
  %21:3 = util.assume.int
      %10<umin = 0, umax = 9007199254740991>,
      %15<umin = 0, umax = 9007199254740991>,
      %20<umin = 0, umax = 9007199254740991>
    : index, index, index
  %22 = iree_tensor_ext.dispatch.workload.ordinal %21#0, 0 : index
  %23 = iree_tensor_ext.dispatch.workload.ordinal %21#1, 1 : index
  %24 = iree_tensor_ext.dispatch.workload.ordinal %21#2, 2 : index
  %25 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%22, %23, %24}
  %26 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%22, %23, %24}
  %27 = iree_tensor_ext.dispatch.tensor.load %25, offsets = [0, 0, 0], sizes = [%22, %23, %24], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf32>>{%22, %23, %24} -> tensor<?x?x?xf32>
  %28 = tensor.empty(%22, %23, %24) : tensor<?x?x?xf32>
  %dim = tensor.dim %27, %c0 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %27, %c1 : tensor<?x?x?xf32>
  %29 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
  %30 = linalg.fill ins(%cst_0 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %31 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%27 : tensor<?x?x?xf32>) outs(%30 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %35 = arith.maxnumf %in, %out : f32
    linalg.yield %35 : f32
  } -> tensor<?x?xf32>
  %32 = linalg.fill ins(%cst : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %33 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%27, %31 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%32 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %35 = arith.subf %in, %in_2 : f32
    %36 = math.exp %35 : f32
    %37 = arith.addf %36, %out : f32
    linalg.yield %37 : f32
  } -> tensor<?x?xf32>
  %34 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %31, %33 : tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%28 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
    %35 = arith.subf %in, %in_2 : f32
    %36 = math.exp %35 : f32
    %37 = arith.divf %36, %in_3 : f32
    linalg.yield %37 : f32
  } -> tensor<?x?x?xf32>
  iree_tensor_ext.dispatch.tensor.store %34, %26, offsets = [0, 0, 0], sizes = [%22, %23, %24], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?xf32>>{%22, %23, %24}
  return
}
// CHECK-LABEL: func.func @softmax_dynamic_with_assume_int_hints(

// -----

// Verify that stack buffer is not created in accumulating GEMMs dispatches;
// it direct writes the result into the destination buffer.

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>]>
func.func @matmul_accumulate_from_readonly(%M: index, %N: index, %K: index) attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N}
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K} -> tensor<?x?xf32>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N} -> tensor<?x?xf32>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N} -> tensor<?x?xf32>
  %7 = linalg.matmul ins(%4, %5 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %3, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
  return
}
// CHECK-LABEL: func.func @matmul_accumulate_from_readonly(
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     linalg.generic
