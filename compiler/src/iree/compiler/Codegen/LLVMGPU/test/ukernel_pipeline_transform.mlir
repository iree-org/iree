// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.abbr_target<hip:"gfx1100">, ukernels = "argmax"}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @argmax_1d_f16i64() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0xFC00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = arith.extui %0 : i32 to i64
    %3 = arith.extui %1 : i32 to i64
    %4 = arith.shli %3, %c32_i64 : i64
    %5 = arith.ori %2, %4 : i64
    %6 = arith.index_castui %5 : i64 to index
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<i64>>
    %8 = flow.dispatch.workload.ordinal %6, 0 : index
    %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf16>>{%8}
    %10 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf16>>{%8} -> tensor<?xf16>
    %11 = tensor.empty() : tensor<i64>
    %12 = tensor.empty() : tensor<f16>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%11 : tensor<i64>) -> tensor<i64>
    %14 = linalg.fill ins(%cst : f16) outs(%12 : tensor<f16>) -> tensor<f16>
    %15:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%10 : tensor<?xf16>) outs(%14, %13 : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_0: i64):
      %16 = linalg.index 0 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.maximumf %in, %out : f16
      %19 = arith.cmpf ogt, %in, %out : f16
      %20 = arith.select %19, %17, %out_0 : i64
      linalg.yield %18, %20 : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    flow.dispatch.tensor.store %15#1, %7, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
    return
  }
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDefault workgroup_size = [32, 1, 1]>
//       CHECK: func.func @argmax_1d_f16i64()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_codegen.ukernel.generic  "__iree_uk_rocm_argmax_F16I64"

// -----
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.abbr_target<hip:"gfx1100">, ukernels = "argmax"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @argmax_2d_f32i64() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = arith.extui %0 : i32 to i64
    %3 = arith.extui %1 : i32 to i64
    %4 = arith.shli %3, %c32_i64 : i64
    %5 = arith.ori %2, %4 : i64
    %6 = arith.index_castui %5 : i64 to index
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16xi64>>
    %8 = flow.dispatch.workload.ordinal %6, 0 : index
    %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x?xf32>>{%8}
    %10 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [16, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x?xf32>>{%8} -> tensor<16x?xf32>
    %11 = tensor.empty() : tensor<16xi64>
    %12 = tensor.empty() : tensor<16xf32>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%11 : tensor<16xi64>) -> tensor<16xi64>
    %14 = linalg.fill ins(%cst : f32) outs(%12 : tensor<16xf32>) -> tensor<16xf32>
    %15:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<16x?xf32>) outs(%14, %13 : tensor<16xf32>, tensor<16xi64>) {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %16 = linalg.index 1 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.maximumf %in, %out : f32
      %19 = arith.cmpf ogt, %in, %out : f32
      %20 = arith.select %19, %17, %out_0 : i64
      linalg.yield %18, %20 : f32, i64
    } -> (tensor<16xf32>, tensor<16xi64>)
    flow.dispatch.tensor.store %15#1, %7, offsets = [0], sizes = [16], strides = [1] : tensor<16xi64> -> !flow.dispatch.tensor<writeonly:tensor<16xi64>>
    return
  }
}

//      CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDefault workgroup_size = [32, 1, 1]>
//      CHECK: func.func @argmax_2d_f32i64
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   %[[SUBVIEW:.*]] = memref.subview{{.*}} memref<16x?xf32
// CHECK-SAME:        to memref<1x?xf32
//      CHECK:   iree_codegen.ukernel.generic  "__iree_uk_rocm_argmax_F32I64" ins(%[[SUBVIEW]]

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.abbr_target<hip:"gfx1100">}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @no_ukernel_argmax_1d_f16i64() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0xFC00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = arith.extui %0 : i32 to i64
    %3 = arith.extui %1 : i32 to i64
    %4 = arith.shli %3, %c32_i64 : i64
    %5 = arith.ori %2, %4 : i64
    %6 = arith.index_castui %5 : i64 to index
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<i64>>
    %8 = flow.dispatch.workload.ordinal %6, 0 : index
    %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf16>>{%8}
    %10 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf16>>{%8} -> tensor<?xf16>
    %11 = tensor.empty() : tensor<i64>
    %12 = tensor.empty() : tensor<f16>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%11 : tensor<i64>) -> tensor<i64>
    %14 = linalg.fill ins(%cst : f16) outs(%12 : tensor<f16>) -> tensor<f16>
    %15:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%10 : tensor<?xf16>) outs(%14, %13 : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_0: i64):
      %16 = linalg.index 0 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.maximumf %in, %out : f16
      %19 = arith.cmpf ogt, %in, %out : f16
      %20 = arith.select %19, %17, %out_0 : i64
      linalg.yield %18, %20 : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    flow.dispatch.tensor.store %15#1, %7, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
    return
  }
}

//      CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute workgroup_size = [1, 1, 1]>
//      CHECK: func.func @no_ukernel_argmax_1d_f16i64()
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//  CHECK-NOT:   iree_codegen.ukernel.generic

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.abbr_target<hip:"gfx1100">, ukernels = "argmax"}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @not_neg_inf_init_argmax_1d() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = arith.extui %0 : i32 to i64
    %3 = arith.extui %1 : i32 to i64
    %4 = arith.shli %3, %c32_i64 : i64
    %5 = arith.ori %2, %4 : i64
    %6 = arith.index_castui %5 : i64 to index
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<i64>>
    %8 = flow.dispatch.workload.ordinal %6, 0 : index
    %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?xf16>>{%8}
    %10 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf16>>{%8} -> tensor<?xf16>
    %11 = tensor.empty() : tensor<i64>
    %12 = tensor.empty() : tensor<f16>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%11 : tensor<i64>) -> tensor<i64>
    %14 = linalg.fill ins(%cst : f16) outs(%12 : tensor<f16>) -> tensor<f16>
    %15:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%10 : tensor<?xf16>) outs(%14, %13 : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_0: i64):
      %16 = linalg.index 0 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = arith.maximumf %in, %out : f16
      %19 = arith.cmpf ogt, %in, %out : f16
      %20 = arith.select %19, %17, %out_0 : i64
      linalg.yield %18, %20 : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    flow.dispatch.tensor.store %15#1, %7, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
    return
  }
}

//      CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUDistribute workgroup_size = [1, 1, 1]>
//      CHECK: func.func @not_neg_inf_init_argmax_1d()
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
//  CHECK-NOT:   iree_codegen.ukernel.generic
