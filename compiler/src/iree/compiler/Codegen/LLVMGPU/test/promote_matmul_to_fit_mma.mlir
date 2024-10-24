// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-promote-matmul-to-fit-mma))"  %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * -64 + 968, 64)>
#map3 = affine_map<()[s0] -> (s0 * -128 + 1281, 128)>
#map4 = affine_map<()[s0] -> (-s0 + 64)>
#map5 = affine_map<()[s0] -> (-s0 + 128)>
#config = #iree_gpu.lowering_config<{workgroup = [1, 16, 16, 0], reduction = [0, 0, 0, 16]}>
func.func @batch_matmul_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
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
  %11 = linalg.batch_matmul {lowering_config = #config} ins(%8, %9 : tensor<1x?x1281xf16>, tensor<1x1281x?xf16>) outs(%10 : tensor<1x?x?xf16>) -> tensor<1x?x?xf16>
  flow.dispatch.tensor.store %11, %2, offsets = [%workgroup_id_z, %3, %4], sizes = [1, %5, %6], strides = [1, 1, 1] : tensor<1x?x?xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
  return
}
// CHECK-LABEL:     func.func @batch_matmul_f16
// CHECK:             %[[LHS_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x968x1281xf16>>
// CHECK:             %[[RHS_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1281x1281xf16>>
// CHECK:             %[[OUT_HANDLE:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x968x1281xf16>>
// CHECK-DAG:         %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_HANDLE]]
// CHECK-DAG:         %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_HANDLE]]
// CHECK:        %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]]
// CHECK:        } : tensor<1x?x1281xf16> to tensor<1x64x1296xf16>
// CHECK:        %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]]
// CHECK:        } : tensor<1x1281x?xf16> to tensor<1x1296x128xf16>
// CHECK:        %[[FILL:.+]] = linalg.fill
// CHECK:        %[[GEMM:.+]] = linalg.batch_matmul
// CHECK-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]]
// CHECK-SAME:     outs(%[[FILL]]

// CHECK:             %[[OUT_SLICE:.+]] = tensor.extract_slice %[[GEMM]]
// CHECK:             flow.dispatch.tensor.store %[[OUT_SLICE]], %[[OUT_HANDLE]]
