// RUN: iree-opt --iree-workgroup-swizzle='logTile=3' %s | FileCheck %s

func.func @matmul() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c96 = arith.constant 96 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x4096xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4096x96xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x96xf32>>
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

//    CHECK-LABEL: func.func @matmul
//          CHECK: %[[WORKGROUPIDX:.*]] = hal.interface.workgroup.id[0] : index
//          CHECK: %[[WORKGROUPIDY:.*]] = hal.interface.workgroup.id[1] : index
//          CHECK: %[[WORKGROUPCOUNTX:.*]] = hal.interface.workgroup.count[0] : index
//          CHECK: %[[WORKGROUPCOUNTY:.*]] = hal.interface.workgroup.count[1] : index
//          CHECK: %[[CST0:.*]] = arith.constant 0 : index
//          CHECK: %[[CST8:.*]] = arith.constant 8 : index
//          CHECK: %[[S0:.*]] = arith.remui %[[WORKGROUPIDY]], %[[CST8]] : index
//          CHECK: %[[S1:.*]] = arith.divui %[[WORKGROUPIDY]], %[[CST8]] : index
//          CHECK: %[[S2:.*]] = arith.muli %[[S0]], %[[WORKGROUPCOUNTX]] : index
//          CHECK: %[[S3:.*]] = arith.addi %[[WORKGROUPIDX]], %[[S2]] : index
//          CHECK: %[[S4:.*]] = arith.remui %[[S3]], %[[CST8]] : index
//          CHECK: %[[S5:.*]] = arith.muli %[[S1]], %[[CST8]] : index
//          CHECK: %[[S6:.*]] = arith.divui %[[S3]], %[[CST8]] : index
//          CHECK: %[[S7:.*]] = arith.addi %[[S4]], %[[S5]] : index
//          CHECK: %[[S8:.*]] = arith.remui %[[WORKGROUPCOUNTY]], %[[CST8]] : index
//          CHECK: %[[S9:.*]] = arith.addi %[[S5]], %[[CST8]] : index
//          CHECK: %[[S10:.*]] = arith.cmpi ne, %[[S8]], %[[CST0]] : index
//          CHECK: %[[S11:.*]] = arith.cmpi ugt, %[[S9]], %[[WORKGROUPCOUNTY]] : index
//          CHECK: %[[S12:.*]] = arith.andi %[[S10]], %[[S11]] : i1
//          CHECK: %[[S13:.*]] = arith.select %[[S12]], %[[WORKGROUPIDX]], %[[S6]] : index
//          CHECK: %[[S14:.*]] = arith.select %[[S12]], %[[WORKGROUPIDY]], %[[S7]] : index



