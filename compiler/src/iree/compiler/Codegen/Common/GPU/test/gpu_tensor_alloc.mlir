// RUN: iree-opt %s --allow-unregistered-dialect --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-tensor-tile-to-serial-loops,iree-codegen-gpu-tensor-alloc))" | FileCheck %s
// RUN: iree-opt %s --allow-unregistered-dialect --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-tensor-tile-to-serial-loops{coalesce-loops},iree-codegen-gpu-tensor-alloc))" | FileCheck %s --check-prefix=COALESCE_LOOPS

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_2048x512x1024() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>> -> tensor<32x128xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [32, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>> -> tensor<32x1024xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [1024, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x128xf32>
  %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%cst : f32) outs(%5 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %9 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%6, %7 : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
  flow.dispatch.tensor.store %9, %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : tensor<32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  return
}

//    CHECK-LABEL: func.func @matmul_2048x512x1024
//         CHECK:    scf.for {{.*}} -> (tensor<32x128xf32>) {
//         CHECK:      %[[A:.*]] = tensor.extract_slice %{{.*}}[0, %{{.*}}] [32, 32] [1, 1] : tensor<32x1024xf32> to tensor<32x32xf32>
//         CHECK:      %[[B:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [32, 128] [1, 1] : tensor<1024x128xf32> to tensor<32x128xf32>
//         CHECK:      %[[PA:.*]] = bufferization.alloc_tensor() copy(%[[A]]) : tensor<32x32xf32>
//         CHECK:      %[[PB:.*]] = bufferization.alloc_tensor() copy(%[[B]]) : tensor<32x128xf32>
//         CHECK:      %[[M:.*]] = linalg.matmul {{.*}} ins(%[[PA]], %[[PB]] : tensor<32x32xf32>, tensor<32x128xf32>) outs(%{{.*}} : tensor<32x128xf32>) -> tensor<32x128xf32>
//         CHECK:      scf.yield %[[M]] : tensor<32x128xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_1x384x384() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384x384xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x384xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x384xf32>> -> tensor<1x384xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [0, %4], sizes = [1, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x384xf32>> -> tensor<1x128xf32>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [384, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x384xf32>> -> tensor<384x128xf32>
  %7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 128, 8]]>} ins(%cst : f32) outs(%5 : tensor<1x128xf32>) -> tensor<1x128xf32>
  %8 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 128, 8]]>} ins(%3, %6 : tensor<1x384xf32>, tensor<384x128xf32>) outs(%7 : tensor<1x128xf32>) -> tensor<1x128xf32>
  flow.dispatch.tensor.store %8, %2, offsets = [0, %4], sizes = [1, 128], strides = [1, 1] : tensor<1x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x384xf32>>
  return
}

//    CHECK-LABEL: func.func @matmul_1x384x384
//      CHECK-NOT:   bufferization.alloc_tensor()
//          CHECK: return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_multi_uses() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>> -> tensor<32x128xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [32, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>> -> tensor<32x1024xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [1024, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x128xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %9 = linalg.matmul ins(%6, %7 : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
  "some_use"(%6) : (tensor<32x1024xf32>) -> ()
  flow.dispatch.tensor.store %9, %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : tensor<32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  return
}

// test corner case where the promoted value has multiple uses.
//    CHECK-LABEL: func.func @matmul_multi_uses
//         CHECK:    %[[C:.*]] = flow.dispatch.tensor.load {{.+}} -> tensor<32x128xf32>
//         CHECK:    %[[A:.*]] = flow.dispatch.tensor.load {{.+}} -> tensor<32x1024xf32>
//         CHECK:    %[[B:.*]] = flow.dispatch.tensor.load {{.+}} -> tensor<1024x128xf32>
//         CHECK:    %[[PA:.*]] = bufferization.alloc_tensor() copy(%[[A]]) : tensor<32x1024xf32>
//         CHECK:    %[[PB:.*]] = bufferization.alloc_tensor() copy(%[[B]]) : tensor<1024x128xf32>
//         CHECK:    %[[M:.*]] = linalg.matmul ins(%[[PA]], %[[PB]] : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%{{.*}} : tensor<32x128xf32>) -> tensor<32x128xf32>
//         CHECK:    "some_use"(%[[A]]) : (tensor<32x1024xf32>) -> ()

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_33x33x903168_f32() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 9.031680e+05 : f32
  %cst_1 = arith.constant 0.949999988 : f32
  %c32 = arith.constant 32 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %0 = affine.min affine_map<()[s0] -> (s0 * -32 + 33, 32)>()[%workgroup_id_x]
  %1 = arith.cmpi eq, %0, %c32 : index
  scf.if %1 {
    %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
    %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
    %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
    %5 = arith.index_castui %2 {stream.alignment = 4096 : index, stream.values = [1240289280 : index, 1789415424 : index]} : i32 to index
    %6 = arith.index_castui %3 {stream.alignment = 8192 : index, stream.values = [633077760 : index, 752295936 : index]} : i32 to index
    %7 = arith.index_castui %4 {stream.alignment = 64 : index, stream.values = [1486349952 : index, 1486358464 : index]} : i32 to index
    %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<33x903168xf32>>
    %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<903168x33xf32>>
    %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<33x33xf32>>
    %11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<33x33xf32>>
    %12 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %13 = flow.dispatch.tensor.load %11, offsets = [%12, 0], sizes = [32, 33], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<33x33xf32>> -> tensor<32x33xf32>
    %14 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [903168, 33], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<903168x33xf32>> -> tensor<903168x33xf32>
    %15 = flow.dispatch.tensor.load %10, offsets = [%12, 0], sizes = [32, 33], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x33xf32>> -> tensor<32x33xf32>
    %16 = flow.dispatch.tensor.load %8, offsets = [%12, 0], sizes = [32, 903168], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x903168xf32>> -> tensor<32x903168xf32>
    %17 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%cst : f32) outs(%13 : tensor<32x33xf32>) -> tensor<32x33xf32>
    %18 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%16, %14 : tensor<32x903168xf32>, tensor<903168x33xf32>) outs(%17 : tensor<32x33xf32>) -> tensor<32x33xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<32x33xf32>) outs(%18 : tensor<32x33xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.divf %out, %cst_0 : f32
      %21 = arith.mulf %in, %cst_1 : f32
      %22 = arith.addf %21, %20 : f32
      linalg.yield %22 : f32
    } -> tensor<32x33xf32>
    flow.dispatch.tensor.store %19, %11, offsets = [%12, 0], sizes = [32, 33], strides = [1, 1] : tensor<32x33xf32> -> !flow.dispatch.tensor<writeonly:tensor<33x33xf32>>
  }
  return
}

// The allocation should not happen when there is any unaligned size, e.g., 33 in this case.
//
// CHECK-LABEL: func.func @matmul_33x33x903168_f32
// CHECK-NOT: bufferization.alloc_tensor()

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @weight_dequant_matmul() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<86x128x2048xi4>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<86x2048xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<86x2048xi4>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xf32>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %7 = flow.dispatch.tensor.load %4, offsets = [%5, %6], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<4096x2048xf32>> -> tensor<32x128xf32>
  %8 = flow.dispatch.tensor.load %3, offsets = [%5, 0, 0], sizes = [32, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xf32>> -> tensor<32x86x128xf32>
  %9 = flow.dispatch.tensor.load %0, offsets = [0, 0, %6], sizes = [86, 128, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128x2048xi4>> -> tensor<86x128x128xi4>
  %10 = flow.dispatch.tensor.load %1, offsets = [0, %6], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x2048xf32>> -> tensor<86x128xf32>
  %11 = flow.dispatch.tensor.load %2, offsets = [0, %6], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x2048xi4>> -> tensor<86x128xi4>
  %12 = tensor.empty() : tensor<86x128x128xf32>
  %13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%9, %10, %11 : tensor<86x128x128xi4>, tensor<86x128xf32>, tensor<86x128xi4>) outs(%12 : tensor<86x128x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: i4, %out: f32):
    %16 = arith.extsi %in : i4 to i32
    %17 = arith.extsi %in_1 : i4 to i32
    %18 = arith.subi %16, %17 : i32
    %19 = arith.sitofp %18 : i32 to f32
    %20 = arith.mulf %19, %in_0 : f32
    linalg.yield %20 : f32
  } -> tensor<86x128x128xf32>
  %14 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %15 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%8, %13 : tensor<32x86x128xf32>, tensor<86x128x128xf32>) outs(%14 : tensor<32x128xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 1, 32]]>} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %16 = arith.mulf %in, %in_0 : f32
    %17 = arith.addf %out, %16 : f32
    linalg.yield %17 : f32
  } -> tensor<32x128xf32>
  flow.dispatch.tensor.store %15, %4, offsets = [%5, %6], sizes = [32, 128], strides = [1, 1] : tensor<32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
  return
}

// CHECK-LABEL: func.func @weight_dequant_matmul()
//       CHECK:   %[[LHS_LD:.+]] = flow.dispatch.tensor.load {{.+}} : !flow.dispatch.tensor<readonly:tensor<4096x86x128xf32>> -> tensor<32x86x128xf32>
// Check that the linalg.fill as the matmul initial result is not fused in the serial loops.
//       CHECK:   %[[FILL:.+]] = linalg.fill
// Check that two serial loops are materialized for reductions.
//       CHECK:   scf.for %{{.+}} = %c0 to %c86 step %c1 iter_args(%[[ARG1:.+]] = %[[FILL]]) -> (tensor<32x128xf32>)
//       CHECK:     scf.for %{{.+}} = %c0 to %c128 step %c32 iter_args(%[[ARG2:.+]] = %[[ARG1]]) -> (tensor<32x128xf32>)
//       CHECK:       %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS_LD]]
// Check that we have a bufferization.alloc_tensor() for in-place bufferization later.
//       CHECK:       %[[RHS_ALLOC:.+]] = bufferization.alloc_tensor() : tensor<1x32x128xf32>
// Check that the weight dequant linalg.generic is fused inside the serial loops.
//       CHECK:       %[[RHS:.+]] = linalg.generic
//  CHECK-SAME:        outs(%[[RHS_ALLOC]] : tensor<1x32x128xf32>)
//       CHECK:       %[[LHS_ALLOC:.+]] = bufferization.alloc_tensor() copy(%[[LHS_SLICE]]) : tensor<32x1x32xf32>
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[LHS_ALLOC]], %[[RHS]] : tensor<32x1x32xf32>, tensor<1x32x128xf32>)
//  CHECK-SAME:         outs(%[[ARG2]] : tensor<32x128xf32>)
//       CHECK:       scf.yield
//       CHECK:     scf.yield

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @conv() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
  %4 = flow.dispatch.tensor.load %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>> -> tensor<1x1x32x256xf32>
  %5 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_z, %workgroup_id_y, 0, 0], sizes = [1, 3, 34, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x1280xf16>> -> tensor<1x3x34x1280xf16>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %3], sizes = [3, 3, 1280, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>> -> tensor<3x3x1280x256xf16>
  %7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>
  %8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>
  flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
  return
}
// Check loops coalescing works
//     COALESCE_LOOPS: func.func @conv()
//     COALESCE_LOOPS:   scf.for
// COALESCE_LOOPS-NOT:   scf.for
//     COALESCE_LOOPS:   return

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 0, 0, 32, 0]]>

func.func @online_attention(%query: tensor<192x1024x64xf16>,
                         %key: tensor<192x1024x64xf16>,
                         %value: tensor<192x1024x64xf16>,
                         %output: tensor<192x1024x64xf32>,
                         %max: tensor<192x1024xf32>,
                         %sum: tensor<192x1024xf32>)
                         -> (tensor<192x1024x64xf32>, tensor<192x1024xf32>) {
  %scale = arith.constant 1.0 : f16

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR],
          lowering_config = #config }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, f16)
        outs(%output, %max, %sum : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
          ^bb0(%score : f32):
            iree_linalg_ext.yield %score : f32
        }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0, %out#2 : tensor<192x1024x64xf32>, tensor<192x1024xf32>
}

// Just check if the operation gets tiled. The actual tiling verification tests
// are in online_attention tiling interface tests.
// CHECK-LABEL: func.func @online_attention
// CHECK: scf.for
// CHECK:   iree_linalg_ext.online_attention
// CHECK: scf.yield
