// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_10_8x8_pattern() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<512x1024xf32>) outs(%3 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1024x512xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>>
  return
}

//    CHECK-LABEL: func.func @transpose_10_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @transpose_021_8x8_pattern() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128x96xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 128, 96], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128x96xf32>> -> tensor<64x128x96xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x96x128xf32>) outs(%3 : tensor<64x128x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<64x128x96xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 128, 96], strides = [1, 1, 1] : tensor<64x128x96xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128x96xf32>>
  return
}

//    CHECK-LABEL: func.func @transpose_021_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @transpose_201_8x8_pattern() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x64x96xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [128, 64, 96], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x64x96xf32>> -> tensor<128x64x96xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x96x128xf32>) outs(%3 : tensor<128x64x96xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<128x64x96xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [128, 64, 96], strides = [1, 1, 1] : tensor<128x64x96xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x64x96xf32>>
  return
}

//   CHECK-LABEL: func.func @transpose_201_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d2, d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @transpose_210_8x8_pattern() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x96x64xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [128, 96, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x96x64xf32>> -> tensor<128x96x64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x96x128xf32>) outs(%3 : tensor<128x96x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<128x96x64xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [128, 96, 64], strides = [1, 1, 1] : tensor<128x96x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x96x64xf32>>
  return
}

//   CHECK-LABEL: func.func @transpose_210_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @transpose_120_8x8_pattern() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x128x64xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [96, 128, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x128x64xf32>> -> tensor<96x128x64xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x96x128xf32>) outs(%3 : tensor<96x128x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<96x128x64xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [96, 128, 64], strides = [1, 1, 1] : tensor<96x128x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x128x64xf32>>
  return
}

//   CHECK-LABEL: func.func @transpose_120_8x8_pattern
//  CHECK-COUNT-8: vector.load
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
// CHECK-COUNT-12: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//  CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
//  CHECK-COUNT-8: vector.shuffle {{.*}} : vector<8xf32>, vector<8xf32>
//      CHECK-NOT: vector.extract
//      CHECK-NOT: vector.insert
//  CHECK-COUNT-8: vector.store

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx2", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @transpose_102() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x64x128xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 96, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x96x128xf32>> -> tensor<64x96x128xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [96, 64, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x64x128xf32>> -> tensor<96x64x128xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x96x128xf32>) outs(%3 : tensor<96x64x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<96x64x128xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [96, 64, 128], strides = [1, 1, 1] : tensor<96x64x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x64x128xf32>>
  return
}

// CHECK-LABEL: func.func @transpose_102
//   CHECK-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
//   CHECK-NOT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 32 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_no_avx2_feature() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x1024xf32>> -> tensor<512x1024xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<512x1024xf32>) outs(%3 : tensor<1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1024x512xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : tensor<1024x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x512xf32>>
  return
}

// CHECK-LABEL: func.func @test_no_avx2_feature
//   CHECK-NOT: vector.shuffle %{{.*}}, %{{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
//   CHECK-NOT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" %{{.*}}, %{{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
