// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-test-vector-distribution-on-reduction \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=CHECK-REDUCTION

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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x1x64xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x1x64xf16>> -> tensor<20x1x64xf16>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
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
  flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : tensor<20x1x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
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
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x32x10x4096xf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 4096], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x10x4096xf16>> -> tensor<2x32x10x4096xf16>
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
    flow.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [2, 32], strides = [1, 1] : tensor<2x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32xf32>>
    return
}
// CHECK-REDUCTION:      #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute

// CHECK-REDUCTION-LABEL: func.func @reduction_with_no_consumer
// CHECK-REDUCTION:           lowering_config = #iree_gpu.lowering_config
// CHECK-REDUCTION-SAME:      partial_reduction = [0, 0, 1, 4096]
// CHECK-REDUCTION-SAME:      subgroup_basis = {{\[}}[1, 1, 1, 8], [0, 1, 2, 3]
// CHECK-REDUCTION-SAME:      thread = [0, 0, 1, 8], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]
// CHECK-REDUCTION-SAME:      workgroup = [1, 1, 0, 0]

// -----

func.func @test_multiple_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.638400e+05 : f32
  %cst_1 = arith.constant 9.99999974E-6 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<writeonly:tensor<2x32x10x16384xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>> -> tensor<2x32x10x16384xf16>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>> -> tensor<2x32x10x16384xf16>
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
  flow.dispatch.tensor.store %12, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : tensor<2x32x10x16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x10x16384xf32>>
  return
}

// Attaches lowering config to the operation with reduction iterator or
// parallel iterator with new dimensions.

// CHECK-REDUCTION-LABEL: func.func @test_multiple_reduction
// CHECK-REDUCTION:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1],
// CHECK-REDUCTION-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
// CHECK-REDUCTION-SAME:    ins(%{{.*}} : tensor<2x32x10x16384xf32>)
// CHECK-REDUCTION-SAME:    outs({{.*}}: tensor<2x32xf32>)
// CHECK-REDUCTION-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-REDUCTION-SAME:               partial_reduction = [0, 0, 1, 8192],
// CHECK-REDUCTION-SAME:               subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-REDUCTION-SAME:               thread = [0, 0, 1, 8], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK-REDUCTION-SAME:               workgroup = [1, 1, 0, 0]
// CHECK-REDUCTION:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1, #map1],
// CHECK-REDUCTION-SAME:    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
// CHECK-REDUCTION-SAME:    ins{{.*}}, {{.*}} : tensor<2x32x10x16384xf32>, tensor<2x32xf32>)
// CHECK-REDUCTION-SAME:    outs(%{{.*}} : tensor<2x32xf32>)
// CHECK-REDUCTION-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-REDUCTION-SAME:              partial_reduction = [0, 0, 1, 8192],
// CHECK-REDUCTION-SAME:              subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-REDUCTION-SAME:              thread = [0, 0, 1, 8], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK-REDUCTION-SAME:              workgroup = [1, 1, 0, 0]
// CHECK-REDUCTION:       %{{.*}} = linalg.generic {indexing_maps = [#map, #map1, #map1, #map],
// CHECK-REDUCTION-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-REDUCTION-SAME:    ins({{.*}}, %{{.*}}, {{.*}} : tensor<2x32x10x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>)
// CHECK-REDUCTION-SAME:    outs(%{{.*}} : tensor<2x32x10x16384xf32>)
// CHECK-REDUCTION-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
// CHECK-REDUCTION-SAME:              subgroup_basis = {{\[}}[1, 1, 1, 16], [0, 1, 2, 3]],
// CHECK-REDUCTION-SAME:              thread = [0, 0, 0, 8], thread_basis = {{\[}}[1, 1, 1, 64], [0, 1, 2, 3]],
// CHECK-REDUCTION-SAME:              workgroup = [1, 1, 0, 8192]
