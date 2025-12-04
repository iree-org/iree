// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-reorder-workgroups{strategy=transpose}))" \
// RUN:   --split-input-file %s | FileCheck --check-prefix=TRANSPOSE %s

// Test that workgroup IDs are transposed: the pass remaps (x,y) to effectively swap
// the iteration order for better memory locality.

func.func @matmul() {
  %c128 = arith.constant 128 : index
  %c96 = arith.constant 96 : index
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
      %sum = arith.addi %arg0, %arg1 : index
    }
  }
  return
}

// TRANSPOSE-LABEL: func.func @matmul
// TRANSPOSE-DAG:     %[[WG_X:.*]] = hal.interface.workgroup.id[0] : index
// TRANSPOSE-DAG:     %[[WG_Y:.*]] = hal.interface.workgroup.id[1] : index
// TRANSPOSE-DAG:     %[[WG_CNT_X:.*]] = hal.interface.workgroup.count[0] : index
// TRANSPOSE-DAG:     %[[WG_CNT_Y:.*]] = hal.interface.workgroup.count[1] : index
//      TRANSPOSE:     %[[MUL:.+]] = arith.muli %[[WG_Y]], %[[WG_CNT_X]] : index
// TRANSPOSE-NEXT:     %[[ADD:.+]] = arith.addi %[[MUL]], %[[WG_X]] : index
// TRANSPOSE-NEXT:     %[[DIV:.+]] = arith.divui %[[ADD]], %[[WG_CNT_Y]] : index
// TRANSPOSE-NEXT:     %[[REM:.+]] = arith.remui %[[ADD]], %[[WG_CNT_Y]] : index
//      TRANSPOSE:     affine.apply #{{.+}}()[%[[REM]]]
//      TRANSPOSE:     scf.for
//      TRANSPOSE:       affine.apply #{{.+}}()[%[[DIV]]]
