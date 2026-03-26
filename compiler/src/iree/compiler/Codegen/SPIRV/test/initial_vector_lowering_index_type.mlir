// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-spirv-initial-vector-lowering))' \
// RUN:  | FileCheck %s

// Verify that vector unrolling does not crash on index element types.
// getMemoryVectorSize() must handle types that are not int or float.

// CHECK-LABEL: func.func @transfer_read_index_type
func.func @transfer_read_index_type(%mem: memref<64xindex>) -> vector<4xindex> {
  %c0 = arith.constant 0 : index
  %idx = arith.constant 0 : index
  %0 = vector.transfer_read %mem[%c0], %idx {in_bounds = [true]} : memref<64xindex>, vector<4xindex>
  return %0 : vector<4xindex>
}
