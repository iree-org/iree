// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @aligned_generic_pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %5 = tensor.empty() : tensor<24x512x16x1xf32>
    %6 = tensor.empty() : tensor<384x512xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<512xf32>, tensor<384x512xf32>) outs(%6 : tensor<384x512xf32>) {
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

// CHECK-LABEL:     func.func @aligned_generic_pack
// CHECK:             %[[IN_0:.+]] = vector.broadcast %{{.+}} : vector<16xf32> to vector<16x16xf32>
// CHECK-COUNT-15:    %{{.+}} = vector.insert {{.+}} : vector<16xf32> into vector<16x16xf32>
// CHECK:             %[[IN_1:.+]] = vector.insert {{.+}} : vector<16xf32> into vector<16x16xf32>
// CHECK:             %[[T0:.+]] = arith.addf %[[IN_0]], %[[IN_1]] : vector<16x16xf32>
// CHECK:             %[[T1:.+]] = arith.minimumf %[[T0]], %{{.+}} : vector<16x16xf32>
// CHECK:             %[[T2:.+]] = arith.maximumf %[[T1]], %{{.+}} : vector<16x16xf32>
// CHECK-COUNT-16:    vector.extract %[[T2]]
// CHECK-COUNT-64:    vector.shuffle

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @aligned_unpack_generic() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
    %5 = tensor.empty() : tensor<384x512xf32>
    %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %5 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %unpack : tensor<512xf32>, tensor<384x512xf32>) outs(%5 : tensor<384x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.addf %in, %in_1 : f32
      %8 = arith.minimumf %7, %cst : f32
      %9 = arith.maximumf %8, %cst_0 : f32
      linalg.yield %9 : f32
    } -> tensor<384x512xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : tensor<384x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
    return
  }
}

// CHECK-LABEL:     func.func @aligned_unpack_generic
// CHECK:             %[[SRC:.+]] = hal.interface.binding.subspan {{.*}} : memref<24x32x16x16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK-COUNT-15:        vector.load %[[SRC]]
// CHECK:                 %[[LAST_LOAD:.+]] = vector.load %[[SRC]]
// CHECK:                 %[[IN_0:.+]] = vector.broadcast %{{.+}} : vector<16xf32> to vector<16x16xf32>
// CHECK:                 %[[T0:.+]] = arith.addf %[[IN_0]], %{{.+}} : vector<16x16xf32>
// CHECK:                 %[[T1:.+]] = arith.minimumf %[[T0]], %{{.+}} : vector<16x16xf32>
// CHECK:                 %[[T2:.+]] = arith.maximumf %[[T1]], %{{.+}} : vector<16x16xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 64 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @unaligned_pack() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<383x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [383, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<383x512xf32>> -> tensor<383x512xf32>
    %3 = tensor.empty() : tensor<24x512x16x1xf32>
    %pack = tensor.pack %2 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3 : tensor<383x512xf32> -> tensor<24x512x16x1xf32>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
    return
  }
}

// CHECK-LABEL:     func.func @unaligned_pack
// CHECK-COUNT-16:    vector.maskedload {{.+}} vector<16xf32>
// CHECK-COUNT-64:    vector.shuffle
