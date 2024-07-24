// RUN: iree-opt --pass-pipeline="builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))" --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @pad_only_dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
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

// CHECK-LABEL: func @pad_only_dispatch()
//       CHECK:   %[[INPUT:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x112x112x64xf32
//       CHECK:   %[[OUTPUT:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x114x114x64xf32
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.if
//       CHECK:         %[[OUTPUT_SUBVIEW_IF:.+]] = memref.subview %[[OUTPUT]]
//       CHECK:         linalg.generic
//  CHECK-SAME:             outs(%[[OUTPUT_SUBVIEW_IF]]
//       CHECK:       else
//       CHECK:         %[[INPUT_SUBVIEW:.+]] = memref.subview %[[INPUT]]
//       CHECK:         %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]]
//       CHECK:         scf.for
//       CHECK:           scf.for
//       CHECK:             scf.for
//       CHECK:               %[[OUTPUT_SLICE:.+]] = memref.subview %[[OUTPUT_SUBVIEW]]
//       CHECK:               %[[RESULT_VEC:.+]] = scf.if %{{.+}} -> (vector<4xf32>) {
//       CHECK:                 %[[VEC_LOAD:.+]] = vector.load %[[INPUT_SUBVIEW]]
//       CHECK:                 scf.yield %[[VEC_LOAD]]
//       CHECK:               }
//       CHECK:               %[[DROP_UNIT_OUTPUT_SLICE:.+]] = memref.subview %[[OUTPUT_SLICE]]
//       CHECK:               vector.store %[[RESULT_VEC]], %[[DROP_UNIT_OUTPUT_SLICE]]

// -----
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
module {
  func.func @pad_with_producer_dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c802816 = arith.constant 802816 : index
    %c72545728 = arith.constant 72545728 : index
    %c72676800 = arith.constant 72676800 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.001000e-05 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c802816) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x56x56x256xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c72545728) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x256x128xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c72676800) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x30x30x128xf32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x56x56x256xf32>> -> tensor<1x56x56x256xf32>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 256, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x256x128xf32>> -> tensor<1x1x256x128xf32>
    %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
    %7 = tensor.empty() : tensor<1x28x28x128xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %9 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%4, %5 : tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) outs(%8 : tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<1x28x28x128xf32>, tensor<128xf32>) outs(%7 : tensor<1x28x28x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %11 = arith.addf %in, %in_1 : f32
      linalg.yield %11 : f32
    } -> tensor<1x28x28x128xf32>
    %padded = tensor.pad %10 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x28x28x128xf32> to tensor<1x30x30x128xf32>
    flow.dispatch.tensor.store %padded, %3, offsets = [0, 0, 0, 0], sizes = [1, 30, 30, 128], strides = [1, 1, 1, 1] : tensor<1x30x30x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x30x30x128xf32>>
    return
  }
}

// CHECK-LABEL: func @pad_with_producer_dispatch()
//       CHECK:   %[[INPUT:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x56x56x256xf32
//       CHECK:   %[[FILTER:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x1x256x128xf32
//       CHECK:   %[[BIAS:.+]] = hal.interface.binding.subspan {{.+}} : memref<128xf32
//       CHECK:   %[[OUTPUT:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x30x30x128xf32
//       CHECK:   scf.for
//       CHECK:     scf.for
//       CHECK:       scf.if
//       CHECK:       else
//   CHECK-DAG:         %[[INPUT_SUBVIEW:.+]] = memref.subview %[[INPUT]]
//   CHECK-DAG:         %[[FILTER_SUBVIEW:.+]] = memref.subview %[[FILTER]]
//   CHECK-DAG:         %[[BIAS_SUBVIEW:.+]] = memref.subview %[[BIAS]]
//   CHECK-DAG:         %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]]
//       CHECK:         scf.for
//       CHECK:           scf.for
//   CHECK-DAG:             %[[INPUT_SLICE:.+]] = memref.subview %[[INPUT_SUBVIEW]]
//       CHECK:             %[[ALLOC:.+]] = memref.alloca
//       CHECK:             scf.for
//   CHECK-DAG:               %[[OUTPUT_SLICE:.+]] = memref.subview %[[OUTPUT_SUBVIEW]]
//   CHECK-DAG:               %[[FILTER_SLICE:.+]] = memref.subview %[[FILTER_SUBVIEW]]
//       CHECK:               linalg.fill
//  CHECK-SAME:                   outs(%[[ALLOC]]
//   CHECK-DAG:               %[[CONV_OUTPUT:.+]] = memref.subview %[[ALLOC]]
//       CHECK:               scf.for
//   CHECK-DAG:                 %[[CONV_INPUT:.+]] = memref.subview %[[INPUT_SLICE]]
//   CHECK-DAG:                 %[[CONV_FILTER:.+]] = memref.subview %[[FILTER_SLICE]]
//       CHECK:                 linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:                     ins(%[[CONV_INPUT]], %[[CONV_FILTER]] :
//  CHECK-SAME:                     outs(%[[CONV_OUTPUT]] :
//       CHECK:               %[[BIAS_INPUT:.+]] = memref.subview %[[BIAS_SUBVIEW]]
//       CHECK:               linalg.generic
//  CHECK-SAME:                   ins(%[[ALLOC]], %[[BIAS_INPUT]] :
//  CHECK-SAME:                   outs(%[[ALLOC]]
//       CHECK:               linalg.fill ins(%{{.+}} :   f32) outs(%[[OUTPUT_SLICE]]
//       CHECK:               %[[INTERIOR_SLICE:.+]] = memref.subview %[[OUTPUT_SLICE]]
//       CHECK:               linalg.generic
//  CHECK-SAME:                   ins(%[[ALLOC]] :
//  CHECK-SAME:                   outs(%[[INTERIOR_SLICE]] :

// -----
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
module {
  func.func @pad_consumer_fusion_dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x14x14x256xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x256x256xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1x14x14x256xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x14x14x256xf32>> -> tensor<1x14x14x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 256, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x14x14x256xf32>> -> tensor<1x14x14x256xf32>
    %padded = tensor.pad %3 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst : f32
    } : tensor<1x14x14x256xf32> to tensor<1x16x16x256xf32>
    %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %4 : tensor<1x16x16x256xf32>, tensor<3x3x256x256xf32>) outs(%5 : tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 256], strides = [1, 1, 1, 1] : tensor<1x14x14x256xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x14x14x256xf32>>
    return
  }
}

//   CHECK-LABEL: func @pad_consumer_fusion_dispatch()
//         CHECK:   %[[INPUT:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x14x14x256xf32, #hal.descriptor_type<storage_buffer>>
//         CHECK:   %[[FILTER:.+]] = hal.interface.binding.subspan {{.+}} : memref<3x3x256x256xf32, #hal.descriptor_type<storage_buffer>>
//         CHECK:   %[[OUTPUT:.+]] = hal.interface.binding.subspan {{.+}} : memref<1x14x14x256xf32, #hal.descriptor_type<storage_buffer>>
//     CHECK-DAG:   %[[FILTER_SUBVIEW:.+]] = memref.subview %[[FILTER]]
//     CHECK-DAG:   %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]]
//         CHECK:   scf.for
//         CHECK:     scf.for
//         CHECK:       scf.for
//         CHECK:         %[[OUTPUT_SUBVIEW_0:.+]] = memref.subview %[[OUTPUT_SUBVIEW]]
//         CHECK:         scf.for
//         CHECK:           scf.for
//         CHECK:             scf.for
// CHECK-COUNT-7:               vector.load %[[INPUT]]
// CHECK-COUNT-8:               vector.load %[[FILTER_SUBVIEW]]
// CHECK-COUNT-8:               vector.fma
//         CHECK:               scf.yield
//         CHECK:             scf.yield
//         CHECK:           scf.yield
//         CHECK:         %[[OUTPUT_SUBVIEW_1:.+]] = memref.subview %[[OUTPUT_SUBVIEW_0]]
// CHECK-COUNT-7:         vector.store %{{.+}}, %[[OUTPUT_SUBVIEW_1]]
