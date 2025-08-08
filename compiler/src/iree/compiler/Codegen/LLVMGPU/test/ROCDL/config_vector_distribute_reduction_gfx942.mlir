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
// CHECK-SAME:      lane_basis = {{\[}}[1, 1, 1, 1, 64], [1, 0, 4, 3]{{\]}}
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1, 8], [0, 1, 4, 3]{{\]}}
// CHECK-SAME:      thread = [0, 0, 0, 8]
// CHECK-SAME:  qk_attrs =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      lane_basis = {{\[}}[1, 1, 1, 1, 64], [1, 0, 3, 4]{{\]}}
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 1, 8], [0, 1, 2, 4]{{\]}}
// CHECK-SAME:      thread = [0, 0, 8, 0]
// CHECK-SAME:  lowering_config =
// CHECK-SAME:    #iree_gpu.lowering_config
// CHECK-SAME:      partial_reduction = [0, 0, 0, 512, 0]
// CHECK-SAME:      workgroup = [1, 1, 0, 0, 16]


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
// CHECK-SAME:      lane_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]
// CHECK-SAME:      partial_reduction = [0, 0, 1, 4096]
// CHECK-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 8], [0, 1, 2, 3]
// CHECK-SAME:      thread = [0, 0, 1, 8],
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
// CHECK-SAME:               lane_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK-SAME:               partial_reduction = [0, 0, 1, 8192],
// CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-SAME:               thread = [0, 0, 1, 8],
// CHECK-SAME:               workgroup = [1, 1, 0, 0]
// CHECK:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1, #map1],
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME:    ins{{.*}}, {{.*}} : tensor<2x32x10x16384xf32>, tensor<2x32xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<2x32xf32>)
// CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME:              lane_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK-SAME:              partial_reduction = [0, 0, 1, 8192],
// CHECK-SAME:              subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-SAME:              thread = [0, 0, 1, 8],
// CHECK:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1, #map1, #map],
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:    ins({{.*}}, %{{.*}}, {{.*}} : tensor<2x32x10x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>)
// CHECK-SAME:    outs(%{{.*}} : tensor<2x32x10x16384xf32>)
// CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME:              lane_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]]
// CHECK-SAME:              reduction = [0, 0, 1, 8192],
// CHECK-SAME:              subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-SAME:              thread = [0, 0, 0, 8],

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
  %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%3, %4 : tensor<128x?x32xf8E4M3FNUZ>, tensor<128x?x32x128xf8E4M3FNUZ>) outs(%2 : tensor<128x128xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 1, 32], subgroup_basis = [[1, 1, 1, 1], [0, 1, 2, 3]], thread = [0, 0, 1, 16], lane_basis = [[1, 1, 1, 2], [0, 1, 2, 3]], workgroup = [1, 1, 0, 0]}>} {
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
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [2, 1, 1] subgroup_size = 64
//       CHECK: func.func @test_dyn_reduction
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 1, 1, 2], [0, 1, 2, 3]],
//  CHECK-SAME:               partial_reduction = [0, 0, 1, 32],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1, 1], [0, 1, 2, 3]],
//  CHECK-SAME:               thread = [0, 0, 1, 16],
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
//  CHECK-SAME:               lane_basis = {{\[}}[1, 64], [0, 1]],
//  CHECK-SAME:               reduction = [0, 4096],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 16], [0, 1]],
//  CHECK-SAME:               thread = [0, 4],
//  CHECK-SAME:               workgroup = [1, 0]
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 64], [0, 1]],
//  CHECK-SAME:               partial_reduction = [0, 4096],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 16], [0, 1]],
//  CHECK-SAME:               thread = [0, 4],
//  CHECK-SAME:               workgroup = [1, 0]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
// Test to not add lowering to gather like operation.
func.func @test_gather_config(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xi64>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf32>>) {
  %c2_i64 = arith.constant 2 : i64
  %c0 = arith.constant 0 : index
  %load1 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xi64>> -> tensor<4096xi64>
  %load2 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4096, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x64xf32>> -> tensor<4096x64xf32>
    %0 = tensor.empty() : tensor<4096x64xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%load1 : tensor<4096xi64>) outs(%0 : tensor<4096x64xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4 = linalg.index 0 : index
      %5 = linalg.index 1 : index
      %extracted = tensor.extract %load2[%4, %5] : tensor<4096x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<4096x64xf32>
    %2 = tensor.empty() : tensor<4096xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<4096x64xf32>) outs(%2 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<4096xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf32>>
  return
}
//      CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
//      CHECK: func.func @test_gather_config
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK:   linalg.generic
//  CHECK-NOT:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//      CHECK:    linalg.yield
//      CHECK:   linalg.generic
// CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-SAME:               lane_basis = {{\[}}[1, 64], [0, 1]],
// CHECK-SAME:               partial_reduction = [0, 64],
// CHECK-SAME:               subgroup_basis = {{\[}}[1, 1], [0, 1]],
// CHECK-SAME:               thread = [0, 1],
// CHECK-SAME:               workgroup = [1, 0]

// -----

func.func @split_reduction_config(%arg0 : tensor<?x131072xf32>,
    %result : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %1 = tensor.dim %arg0, %c0 : tensor<?x131072xf32>
  %2 = tensor.empty(%1) : tensor<?x1024xf32>
  %3 = scf.forall (%arg1) = (0) to (131072) step (128) shared_outs(%arg2 = %2) -> (tensor<?x1024xf32>) {
    %4 = tensor.extract_slice %arg0[0, %arg1] [%1, 128] [1, 1] : tensor<?x131072xf32> to tensor<?x128xf32>
    %5 = affine.apply affine_map<()[s0] -> (s0 floordiv 128)>()[%arg1]
    %extracted_slice = tensor.extract_slice %arg2[0, %5] [%1, 1] [1, 1] : tensor<?x1024xf32> to tensor<?xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<?xf32>) -> tensor<?xf32>
    %7 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%4 : tensor<?x128xf32>) outs(%6 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.addf %in, %out : f32
        linalg.yield %8 : f32
    } -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg2[0, %5] [%1, 1] [1, 1] : tensor<?xf32> into tensor<?x1024xf32>
    }
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
  iree_tensor_ext.dispatch.tensor.store %3, %result,
      offsets = [0, 0], sizes = [%1, 1024], strides = [1, 1]
      : tensor<?x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%1}
  return
}
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
//      CHECK: func @split_reduction_config
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #iree_gpu.lowering_config

// -----

#map = affine_map<()[s0] -> (s0 floordiv 4)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @test_store_to_buffer(%arg0: index, %arg1: index) {
  %cst = arith.constant 1.000000e+00 : f32
  %c2_i64 = arith.constant 2 : i64
  %c0 = arith.constant 0 : index
  %0:2 = util.assume.int
      %arg0<umin = 0, umax = 36028797018963964, udiv = 4>,
      %arg1<udiv = 4>
    : index, index
  %1 = iree_tensor_ext.dispatch.workload.ordinal %0#0, 0 : index
  %2 = iree_tensor_ext.dispatch.workload.ordinal %0#1, 1 : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<?x4096xbf16, #hal.descriptor_type<storage_buffer>>{%1}
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%2}
  %5 = affine.apply #map()[%2]
  %6 = tensor.empty(%5) : tensor<?x4x4096xf32>
  %7 = affine.apply #map()[%1]
  %expand_shape = memref.expand_shape %4 [[0, 1]] output_shape [%7, 4] : memref<?xf32, #hal.descriptor_type<storage_buffer>> into memref<?x4xf32, #hal.descriptor_type<storage_buffer>>
  %expand_shape_0 = memref.expand_shape %3 [[0, 1], [2]] output_shape [%7, 4, 4096] : memref<?x4096xbf16, #hal.descriptor_type<storage_buffer>> into memref<?x4x4096xbf16, #hal.descriptor_type<storage_buffer>>
  %8 = iree_codegen.load_from_buffer %expand_shape_0 : memref<?x4x4096xbf16, #hal.descriptor_type<storage_buffer>> -> tensor<?x4x4096xbf16>
  %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<?x4x4096xbf16>) outs(%6 : tensor<?x4x4096xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %13 = arith.extf %in : bf16 to f32
    linalg.yield %13 : f32
  } -> tensor<?x4x4096xf32>
  %10 = tensor.empty(%7) : tensor<?x4xf32>
  %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<?x4xf32>) -> tensor<?x4xf32>
  %12 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9 : tensor<?x4x4096xf32>) outs(%11 : tensor<?x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = math.fpowi %in, %c2_i64 : f32, i64
    %14 = arith.addf %13, %out : f32
    linalg.yield %14 : f32
  } -> tensor<?x4xf32>
  iree_codegen.store_to_buffer %12, %expand_shape : tensor<?x4xf32> into memref<?x4xf32, #hal.descriptor_type<storage_buffer>>
  return
}
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64
// CHECK-LABEL: @test_store_to_buffer
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//   CHECK-NOT:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
//  CHECK-SAME:               partial_reduction = [0, 0, 4096],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 8], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 0, 8],
//  CHECK-SAME:               workgroup = [1, 1, 0]


// -----


#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
// Test that the thread vector size maximizes memory bandwidth of the largest tensor
// operand (tensor<6656x16384xf16>). We want to emit dwordx4 load, so the vector size
// should be 8xf16 ==> 16 bytes.
func.func @batch_matvec_f16_f32() {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16384xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6656x16384xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x6656xf32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 16384], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16384xf16>> -> tensor<4x16384xf16>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [6656, 16384], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6656x16384xf16>> -> tensor<6656x16384xf16>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4, 6656], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x6656xf32>> -> tensor<4x6656xf32>
  %7 = tensor.empty() : tensor<4x6656xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x6656xf32>) -> tensor<4x6656xf32>
  %mv = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ], iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%4, %5 : tensor<4x16384xf16>, tensor<6656x16384xf16>)
  outs(%8 : tensor<4x6656xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %10 = arith.extf %in : f16 to f32
    %11 = arith.extf %in_0 : f16 to f32
    %12 = arith.mulf %10, %11 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<4x6656xf32>
  iree_tensor_ext.dispatch.tensor.store %mv, %2, offsets = [0, 0], sizes = [4, 6656], strides = [1, 1] : tensor<4x6656xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x6656xf32>>
  return
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: @batch_matvec_f16_f32
//       CHECK:   linalg.generic
//  CHECK-SAME:      attrs = {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:                 lane_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
//  CHECK-SAME:                 partial_reduction = [0, 0, 512],
//  CHECK-SAME:                 subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:                 thread = [0, 0, 8],
//  CHECK-SAME:                 workgroup = [2, 1, 0]
