// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=swizzle logTile=3}))" \
// RUN:   --split-input-file %s | FileCheck --check-prefix=SWIZZLE %s

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=transpose}))" \
// RUN:   --split-input-file %s | FileCheck --check-prefix=TRANSPOSE %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @matmul() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c96 = arith.constant 96 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4096x96xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x96xf32>>
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
      %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [32, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x4096xf32>> -> tensor<32x4096xf32>
      %9 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x96xf32>> -> tensor<4096x32xf32>
      %10 = tensor.extract_slice %3[%arg0, %arg1] [32, 32] [1, 1] : tensor<128x96xf32> to tensor<32x32xf32>
      %11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 16]]>} ins(%8, %9 : tensor<32x4096xf32>, tensor<4096x32xf32>) outs(%10 : tensor<32x32xf32>) -> tensor<32x32xf32>
      flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x96xf32>>
    }
  }
  return
}

// SWIZZLE-LABEL: func.func @matmul
// SWIZZLE:         %[[WG_X:.*]] = hal.interface.workgroup.id[0] : index
// SWIZZLE:         %[[WG_Y:.*]] = hal.interface.workgroup.id[1] : index
// SWIZZLE:         %[[WG_CNT_X:.*]] = hal.interface.workgroup.count[0] : index
// SWIZZLE:         %[[WG_CNT_Y:.*]] = hal.interface.workgroup.count[1] : index
// SWIZZLE:         %[[CST0:.*]] = arith.constant 0 : index
// SWIZZLE:         %[[CST8:.*]] = arith.constant 8 : index
// SWIZZLE:         %[[S0:.*]] = arith.remui %[[WG_Y]], %[[CST8]] : index
// SWIZZLE:         %[[S1:.*]] = arith.divui %[[WG_Y]], %[[CST8]] : index
// SWIZZLE:         %[[S2:.*]] = arith.muli %[[S0]], %[[WG_CNT_X]] : index
// SWIZZLE:         %[[S3:.*]] = arith.addi %[[WG_X]], %[[S2]] : index
// SWIZZLE:         %[[S4:.*]] = arith.remui %[[S3]], %[[CST8]] : index
// SWIZZLE:         %[[S5:.*]] = arith.muli %[[S1]], %[[CST8]] : index
// SWIZZLE:         %[[S6:.*]] = arith.divui %[[S3]], %[[CST8]] : index
// SWIZZLE:         %[[S7:.*]] = arith.addi %[[S4]], %[[S5]] : index
// SWIZZLE:         %[[S8:.*]] = arith.remui %[[WG_CNT_Y]], %[[CST8]] : index
// SWIZZLE:         %[[S9:.*]] = arith.addi %[[S5]], %[[CST8]] : index
// SWIZZLE:         %[[S10:.*]] = arith.cmpi ne, %[[S8]], %[[CST0]] : index
// SWIZZLE:         %[[S11:.*]] = arith.cmpi ugt, %[[S9]], %[[WG_CNT_Y]] : index
// SWIZZLE:         %[[S12:.*]] = arith.andi %[[S10]], %[[S11]] : i1
// SWIZZLE:         %[[S13:.*]] = arith.select %[[S12]], %[[WG_X]], %[[S6]] : index
// SWIZZLE:         %[[S14:.*]] = arith.select %[[S12]], %[[WG_Y]], %[[S7]] : index

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
