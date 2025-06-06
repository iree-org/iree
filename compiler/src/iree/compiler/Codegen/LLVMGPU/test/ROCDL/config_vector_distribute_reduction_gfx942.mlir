// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// CHECK:      #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @attention_20x1x64x4096x64() {
  %cst = arith.constant 1.250000e-01 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x1x64xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x1x64xf16>> -> tensor<20x1x64xf16>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %7 = tensor.empty() : tensor<20x1x64xf16>
  %8 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
               affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
               affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
               affine_map<(d0, d1, d2, d3, d4) -> ()>,
               affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
               ins(%4, %5, %6, %cst : tensor<20x1x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x1x64xf16>) {
                ^bb0(%score: f32):
                  iree_linalg_ext.yield %score : f32
               } -> tensor<20x1x64xf16>
  iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : tensor<20x1x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
  return
}

// CHECK:      decomposition_config =
// CHECK-SAME:  pv_attrs =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1, 1], [0, 1, 3, 4]{{\]}}
// CHECK-SAME:      thread = [0, 0, 0, 8]
// CHECK-SAME:      thread_basis = {{\[}}[1, 1, 1, 1, 64], [1, 0, 4, 3]{{\]}}
// CHECK-SAME:  qk_attrs =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1, 1], [0, 1, 2, 3]{{\]}}
// CHECK-SAME:      thread = [0, 0, 8, 0]
// CHECK-SAME:      thread_basis = {{\[}}[1, 1, 1, 1, 64], [1, 0, 3, 4]{{\]}}
// CHECK-SAME:  lowering_config =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      partial_reduction = [0, 0, 0, 64, 0]
// CHECK-SAME:      workgroup = [1, 1, 0, 0, 0]


// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_with_no_consumer() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.096000e+04 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %c69524992 = arith.constant 69524992 : index
    %c74767872 = arith.constant 74767872 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x4096xf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 4096], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x4096xf16>> -> tensor<2x32x10x4096xf16>
    %3 = tensor.empty() : tensor<2x32x10x4096xf32>
    %4 = tensor.empty() : tensor<2x32xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x32x10x4096xf16>) outs(%3 : tensor<2x32x10x4096xf32>) {
    ^bb0(%in: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    linalg.yield %11 : f32
    } -> tensor<2x32x10x4096xf32>
    %6 = linalg.fill  ins(%cst : f32) outs(%4 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5 : tensor<2x32x10x4096xf32>) outs(%6 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
    %11 = arith.addf %in, %out : f32
    linalg.yield %11 : f32
    } -> tensor<2x32xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [2, 32], strides = [1, 1] : tensor<2x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32xf32>>
    return
}
// CHECK:      #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute

// CHECK-LABEL: func.func @reduction_with_no_consumer
// CHECK:           lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:      partial_reduction = [0, 0, 1, 4096]
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]
// CHECK-SAME:      thread = [0, 0, 1, 4], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]
// CHECK-SAME:      workgroup = [1, 1, 0, 0]

// -----

func.func @test_multiple_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.638400e+05 : f32
  %cst_1 = arith.constant 9.99999974E-6 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x10x16384xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>> -> tensor<2x32x10x16384xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>> -> tensor<2x32x10x16384xf16>
  %5 = tensor.empty() : tensor<2x32x10x16384xf32>
  %6 = tensor.empty() : tensor<2x32xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<2x32x10x16384xf16>) outs(%5 : tensor<2x32x10x16384xf32>) {
  ^bb0(%in: f16, %out: f32):
    %13 = arith.extf %in : f16 to f32
    linalg.yield %13 : f32
  } -> tensor<2x32x10x16384xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%7 : tensor<2x32x10x16384xf32>) outs(%8 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = arith.addf %in, %out : f32
    linalg.yield %13 : f32
  } -> tensor<2x32xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<2x32xf32>) outs(%6 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = arith.divf %in, %cst_0 : f32
    linalg.yield %13 : f32
  } -> tensor<2x32xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%7, %10 : tensor<2x32x10x16384xf32>, tensor<2x32xf32>) outs(%8 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %13 = arith.subf %in, %in_2 : f32
    %14 = arith.mulf %13, %13 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<2x32xf32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %10, %11 : tensor<2x32x10x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>) outs(%5 : tensor<2x32x10x16384xf32>) {
  ^bb0(%in: f16, %in_2: f32, %in_3: f32, %out: f32):
    %13 = arith.divf %in_3, %cst_0 : f32
    %14 = arith.addf %13, %cst_1 : f32
    %15 = math.rsqrt %14 : f32
    %16 = arith.extf %in : f16 to f32
    %17 = arith.subf %16, %in_2 : f32
    %18 = arith.mulf %17, %15 : f32
    linalg.yield %18 : f32
  } -> tensor<2x32x10x16384xf32>
  iree_tensor_ext.dispatch.tensor.store %12, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : tensor<2x32x10x16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x10x16384xf32>>
  return
}

// Attaches lowering config to the operation with reduction iterator or
// parallel iterator with new dimensions.

// CHECK-LABEL: func.func @test_multiple_reduction
// CHECK:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1],
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME:    ins(%{{.*}} : tensor<2x32x10x16384xf32>)
// CHECK-SAME:    outs({{.*}}: tensor<2x32xf32>)
// CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME:               partial_reduction = [0, 0, 1, 4096],
// CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-SAME:               thread = [0, 0, 1, 4], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK-SAME:               workgroup = [1, 1, 0, 0]
// CHECK:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1, #map1],
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME:    ins{{.*}}, {{.*}} : tensor<2x32x10x16384xf32>, tensor<2x32xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<2x32xf32>)
// CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME:              partial_reduction = [0, 0, 1, 4096],
// CHECK-SAME:              subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-SAME:              thread = [0, 0, 1, 4], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1, #map1, #map],
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:    ins({{.*}}, %{{.*}}, {{.*}} : tensor<2x32x10x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<2x32x10x16384xf32>)
// CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME:              reduction = [0, 0, 1, 4096],
// CHECK-SAME:              subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-SAME:              thread = [0, 0, 0, 4], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @test_dyn_reduction(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<128x128xf32>>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32xf8E4M3FNUZ>>{%arg1}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32x128xf8E4M3FNUZ>>{%arg1}
  %2 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<128x128xf32>> -> tensor<128x128xf32>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [128, %arg1, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32xf8E4M3FNUZ>>{%arg1} -> tensor<128x?x32xf8E4M3FNUZ>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [128, %arg1, 32, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32x128xf8E4M3FNUZ>>{%arg1} -> tensor<128x?x32x128xf8E4M3FNUZ>
  %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : tensor<128x?x32xf8E4M3FNUZ>, tensor<128x?x32x128xf8E4M3FNUZ>) outs(%2 : tensor<128x128xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 1, 32], subgroup_basis = [[1, 1, 1, 1], [0, 1, 2, 3]], thread = [0, 0, 1, 16], thread_basis = [[1, 1, 1, 2], [0, 1, 2, 3]], workgroup = [1, 1, 0, 0]}>} {
  ^bb0(%in: f8E4M3FNUZ, %in_0: f8E4M3FNUZ, %out: f32):
    %6 = arith.extf %in : f8E4M3FNUZ to f32
    %7 = arith.extf %in_0 : f8E4M3FNUZ to f32
    %8 = arith.mulf %6, %7 : f32
    %9 = arith.addf %out, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<128x128xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<128x128xf32>>
  return
}
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [8, 1, 1] subgroup_size = 64
//       CHECK: func.func @test_dyn_reduction
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               partial_reduction = [0, 0, 1, 32],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1, 1], [0, 1, 2, 3]],
//  CHECK-SAME:               thread = [0, 0, 1, 4], thread_basis = {{\[}}[1, 1, 1, 8], [0, 1, 2, 3]],
//  CHECK-SAME:               workgroup = [1, 1, 0, 0]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @test_multiple_stores(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<f32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xf32>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4096xf32>>) {
  %c2_i64 = arith.constant 2 : i64
  %c0 = arith.constant 0 : index
  %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>> -> tensor<4x4096xf32>
  %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>> -> tensor<4x4096xf32>
  %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
  %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xf32>> -> tensor<4xf32>
  %4 = tensor.empty() : tensor<4x4096xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %2 : tensor<4x4096xf32>, tensor<4x4096xf32>, tensor<f32>) outs(%4 : tensor<4x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %7 = arith.mulf %in_0, %in_1 : f32
    %8 = arith.addf %in, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<4x4096xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<4x4096xf32>) outs(%3 : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %7 = math.fpowi %in, %c2_i64 : f32, i64
    %8 = arith.addf %7, %out : f32
    linalg.yield %8 : f32
  } -> tensor<4xf32>
  iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xf32>>
  iree_tensor_ext.dispatch.tensor.store %5, %arg4, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : tensor<4x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4096xf32>>
  return
}
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [1024, 1, 1] subgroup_size = 64
//       CHECK: func.func @test_multiple_stores
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               reduction = [0, 4096],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 16], [0, 1]],
//  CHECK-SAME:               thread = [0, 4], thread_basis = {{\[}}[1, 64], [0, 1]],
//  CHECK-SAME:               workgroup = [1, 0]
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               partial_reduction = [0, 4096],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 16], [0, 1]],
//  CHECK-SAME:               thread = [0, 4], thread_basis = {{\[}}[1, 64], [0, 1]],
//  CHECK-SAME:               workgroup = [1, 0]
