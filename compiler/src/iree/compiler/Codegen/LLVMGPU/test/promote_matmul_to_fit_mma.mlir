// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-promote-matmul-to-fit-mma{target-dimensions=parallel}))"  %s | FileCheck %s --check-prefixes=ALL,PARALLEL
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-promote-matmul-to-fit-mma{target-dimensions=reduction}))" %s | FileCheck %s --check-prefixes=ALL,REDUCTION

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * -64 + 968, 64)>
#map3 = affine_map<()[s0] -> (s0 * -128 + 1281, 128)>
#map4 = affine_map<()[s0] -> (-s0 + 64)>
#map5 = affine_map<()[s0] -> (-s0 + 128)>
func.func @batch_matmul_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_y]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %4 = affine.apply #map1()[%workgroup_id_x]
  %5 = affine.min #map2()[%workgroup_id_y]
  %6 = affine.min #map3()[%workgroup_id_x]
  %7 = flow.dispatch.tensor.load %2, offsets = [%workgroup_id_z, %3, %4], sizes = [1, %5, %6], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>> -> tensor<1x?x?xf16>
  %8 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_z, %3, 0], sizes = [1, %5, 1281], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>> -> tensor<1x?x1281xf16>
  %9 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_z, 0, %4], sizes = [1, 1281, %6], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>> -> tensor<1x1281x?xf16>
  %10 = linalg.fill ins(%cst : f16) outs(%7 : tensor<1x?x?xf16>) -> tensor<1x?x?xf16>
  %11 = linalg.batch_matmul ins(%8, %9 : tensor<1x?x1281xf16>, tensor<1x1281x?xf16>) outs(%10 : tensor<1x?x?xf16>) -> tensor<1x?x?xf16>
  flow.dispatch.tensor.store %11, %2, offsets = [%workgroup_id_z, %3, %4], sizes = [1, %5, %6], strides = [1, 1, 1] : tensor<1x?x?xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  return
}
// ALL-LABEL:     func.func @batch_matmul_f16
// ALL:             %[[LHS_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
// ALL:             %[[RHS_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
// ALL:             %[[OUT_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
// ALL-DAG:         %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_HANDLE]]
// ALL-DAG:         %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_HANDLE]]
// PARALLEL:        %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]]
// PARALLEL:        } : tensor<1x?x1281xf16> to tensor<1x64x1281xf16>
// PARALLEL:        %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]]
// PARALLEL:        } : tensor<1x1281x?xf16> to tensor<1x1281x128xf16>
// PARALLEL:        %[[FILL:.+]] = linalg.fill
// PARALLEL:        %[[GEMM:.+]] = linalg.batch_matmul
// PARALLEL-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]]
// PARALLEL-SAME:     outs(%[[FILL]]

// The reduction dim is not tiled in the test case, so it pads it to the same
// shape.
// REDUCTION-DAG:   %[[FILL_DEST:.+]] = flow.dispatch.tensor.load %[[OUT_HANDLE]]
// REDUCTION:       %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[FILL_DEST]]
// REDUCTION:       %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]]
// REDUCTION:       } : tensor<1x?x1281xf16> to tensor<1x?x1281xf16>
// REDUCTION:       %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]]
// REDUCTION:       } : tensor<1x1281x?xf16> to tensor<1x1281x?xf16>
// REDUCTION:       %[[GEMM:.+]] = linalg.batch_matmul
// REDUCTION-SAME:    ins(%[[PADDED_LHS]], %[[PADDED_RHS]]
// REDUCTION-SAME:    outs(%[[FILL]]

// ALL:             %[[OUT_SLICE:.+]] = tensor.extract_slice %[[GEMM]]
// ALL:             flow.dispatch.tensor.store %[[OUT_SLICE]], %[[OUT_HANDLE]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * -64 + 968, 64)>
#map3 = affine_map<()[s0] -> (s0 * -128 + 1281, 128)>
#map4 = affine_map<()[s0] -> (-s0 + 64)>
#map5 = affine_map<()[s0] -> (-s0 + 128)>
#map6 = affine_map<(d0) -> (-d0 + 1281, 64)>
func.func @batch_matmul_pad_reduction_after_tiling() {
  %c64 = arith.constant 64 : index
  %c1281 = arith.constant 1281 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_y]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %4 = affine.apply #map1()[%workgroup_id_x]
  %5 = affine.min #map2()[%workgroup_id_y]
  %6 = affine.min #map3()[%workgroup_id_x]
  %7 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_z, %3, 0], sizes = [1, %5, 1281], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>> -> tensor<1x?x1281xf16>
  %dim = tensor.dim %7, %c1 : tensor<1x?x1281xf16>
  %8 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_z, 0, %4], sizes = [1, 1281, %6], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>> -> tensor<1x1281x?xf16>
  %dim_0 = tensor.dim %8, %c2 : tensor<1x1281x?xf16>
  %9 = affine.apply #map4()[%5]
  %padded = tensor.pad %7 low[0, 0, 0] high[0, %9, 0] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f16
  } : tensor<1x?x1281xf16> to tensor<1x64x1281xf16>
  %10 = affine.apply #map5()[%6]
  %padded_2 = tensor.pad %8 low[0, 0, 0] high[0, 0, %10] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f16
  } : tensor<1x1281x?xf16> to tensor<1x1281x128xf16>
  %11 = tensor.empty() : tensor<1x64x128xf16>
  %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
  %13 = scf.for %arg0 = %c0 to %c1281 step %c64 iter_args(%arg1 = %12) -> (tensor<1x64x128xf16>) {
    %14 = affine.min #map6(%arg0)
    %extracted_slice_4 = tensor.extract_slice %padded[0, 0, %arg0] [1, 64, %14] [1, 1, 1] : tensor<1x64x1281xf16> to tensor<1x64x?xf16>
    %extracted_slice_5 = tensor.extract_slice %padded_2[0, %arg0, 0] [1, %14, 128] [1, 1, 1] : tensor<1x1281x128xf16> to tensor<1x?x128xf16>
    %15 = linalg.batch_matmul ins(%extracted_slice_4, %extracted_slice_5 : tensor<1x64x?xf16>, tensor<1x?x128xf16>) outs(%arg1 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
    scf.yield %15 : tensor<1x64x128xf16>
  }
  %extracted_slice_3 = tensor.extract_slice %13[0, 0, 0] [1, %5, %6] [1, 1, 1] : tensor<1x64x128xf16> to tensor<1x?x?xf16>
  flow.dispatch.tensor.store %extracted_slice_3, %2, offsets = [%workgroup_id_z, %3, %4], sizes = [1, %5, %6], strides = [1, 1, 1] : tensor<1x?x?xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  return
}
// The padding on parallel dims is a nop because they are already padded. Skip
// the check for the testcase.
// ALL-LABEL:     func.func @batch_matmul_pad_reduction_after_tiling
// ALL:             %[[LHS_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
// ALL:             %[[RHS_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
// ALL:             %[[OUT_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
// ALL-DAG:         %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_HANDLE]]
// ALL-DAG:         %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_HANDLE]]
// REDUCTION:       %[[INIT:.+]] = tensor.empty() : tensor<1x64x128xf16>
// REDUCTION:       %[[FILL:.+]] = linalg.fill ins(%{{.+}}) outs(%[[INIT]]
// REDUCTION:       %[[RES:.+]] = scf.for {{.+}} iter_args(%[[ITER:.+]] = %[[FILL]])
// REDUCTION:         %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
// REDUCTION:         %[[PADDED_LHS:.+]] = tensor.pad %[[LHS_SLICE]]
// REDUCTION:         } : tensor<1x?x?xf16> to tensor<1x64x64xf16>
// REDUCTION:         %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
// REDUCTION:         %[[PADDED_RHS:.+]] = tensor.pad %[[RHS_SLICE]]
// REDUCTION:         } : tensor<1x?x?xf16> to tensor<1x64x128xf16>
// REDUCTION:         %[[GEMM:.+]] = linalg.batch_matmul
// REDUCTION-SAME:      ins(%[[PADDED_LHS]], %[[PADDED_RHS]]
// REDUCTION-SAME:      outs(%[[ITER]]
// REDUCTION:         scf.yield %[[GEMM]]
// REDUCTION:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[RES]]
// REDUCTION:       flow.dispatch.tensor.store %[[OUT_SLICE]], %[[OUT_HANDLE]]
