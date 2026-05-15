// RUN: iree-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(iree-spirv-initial-vector-lowering))' \
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

// -----

// Verify that vector.step is unrolled to the native vector size.
// getComputeVectorSize(5) == 1 since 5 is not divisible by 4, 3, or 2.

// CHECK-LABEL: func.func @step_unroll
//   CHECK-DAG:   %[[CST4:.+]] = arith.constant dense<4> : vector<1xindex>
//   CHECK-DAG:   %[[CST3:.+]] = arith.constant dense<3> : vector<1xindex>
//   CHECK-DAG:   %[[CST2:.+]] = arith.constant dense<2> : vector<1xindex>
//   CHECK-DAG:   %[[CST1:.+]] = arith.constant dense<1> : vector<1xindex>
//       CHECK:   %[[STEP:.+]] = vector.step : vector<1xindex>
//       CHECK:   %[[INS0:.+]] = vector.insert_strided_slice %[[STEP]], %{{.+}} {offsets = [0], strides = [1]} : vector<1xindex> into vector<5xindex>
//       CHECK:   %[[ADD1:.+]] = arith.addi %[[STEP]], %[[CST1]] : vector<1xindex>
//       CHECK:   %[[INS1:.+]] = vector.insert_strided_slice %[[ADD1]], %[[INS0]] {offsets = [1], strides = [1]} : vector<1xindex> into vector<5xindex>
//       CHECK:   %[[ADD2:.+]] = arith.addi %[[STEP]], %[[CST2]] : vector<1xindex>
//       CHECK:   %[[INS2:.+]] = vector.insert_strided_slice %[[ADD2]], %[[INS1]] {offsets = [2], strides = [1]} : vector<1xindex> into vector<5xindex>
//       CHECK:   %[[ADD3:.+]] = arith.addi %[[STEP]], %[[CST3]] : vector<1xindex>
//       CHECK:   %[[INS3:.+]] = vector.insert_strided_slice %[[ADD3]], %[[INS2]] {offsets = [3], strides = [1]} : vector<1xindex> into vector<5xindex>
//       CHECK:   %[[ADD4:.+]] = arith.addi %[[STEP]], %[[CST4]] : vector<1xindex>
//       CHECK:   %[[INS4:.+]] = vector.insert_strided_slice %[[ADD4]], %[[INS3]] {offsets = [4], strides = [1]} : vector<1xindex> into vector<5xindex>
//       CHECK:   vector.store %[[INS4]], %{{.+}}[%{{.+}}] : memref<5xindex>, vector<5xindex>
func.func @step_unroll(%dest: memref<5xindex>) {
  %c0 = arith.constant 0 : index
  %v0 = vector.step : vector<5xindex>
  vector.store %v0, %dest[%c0] : memref<5xindex>, vector<5xindex>
  return
}
