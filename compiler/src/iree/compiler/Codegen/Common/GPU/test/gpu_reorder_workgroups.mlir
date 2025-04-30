// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=transpose}))" \
// RUN:   --split-input-file %s | FileCheck --check-prefix=TRANSPOSE %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c96 = arith.constant 96 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x96xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x96xf32>>
  %3 = tensor.empty() : tensor<128x96xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
  scf.for %arg0 = %4 to %c128 step %5 {
    %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c96 step %7 {
      %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [32, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x4096xf32>> -> tensor<32x4096xf32>
      %9 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4096, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x96xf32>> -> tensor<4096x32xf32>
      %10 = tensor.extract_slice %3[%arg0, %arg1] [32, 32] [1, 1] : tensor<128x96xf32> to tensor<32x32xf32>
      %11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>} ins(%8, %9 : tensor<32x4096xf32>, tensor<4096x32xf32>) outs(%10 : tensor<32x32xf32>) -> tensor<32x32xf32>
      iree_tensor_ext.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x96xf32>>
    }
  }
  return
}

// TRANSPOSE-LABEL: func.func @matmul
// TRANSPOSE:         %[[WG_X:.*]] = hal.interface.workgroup.id[0] : index
// TRANSPOSE:         %[[WG_Y:.*]] = hal.interface.workgroup.id[1] : index
// TRANSPOSE:         %[[WG_CNT_X:.*]] = hal.interface.workgroup.count[0] : index
// TRANSPOSE:         %[[WG_CNT_Y:.*]] = hal.interface.workgroup.count[1] : index
// TRANSPOSE:         %[[MUL:.+]] = arith.muli %[[WG_Y]], %[[WG_CNT_X]] : index
// TRANSPOSE:         %[[ADD:.+]] = arith.addi %[[MUL]], %[[WG_X]] : index
// TRANSPOSE:         %[[DIV:.+]] = arith.divui %[[ADD]], %[[WG_CNT_Y]] : index
// TRANSPOSE:         %[[REM:.+]] = arith.remui %[[ADD]], %[[WG_CNT_Y]] : index
// TRANSPOSE-DAG:     affine.apply #{{.+}}()[%[[DIV]]]
// TRANSPOSE-DAG:     affine.apply #{{.+}}()[%[[REM]]]
