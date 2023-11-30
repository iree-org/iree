// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-spirv-final-vector-lowering))' %s | FileCheck %s

func.func @create_mask(%idx: index) -> vector<1xi1> {
  %c1 = arith.constant 1 : index
  %0 = vector.create_mask %c1, %idx, %c1 : vector<1x1x1xi1>
  %1 = vector.shape_cast %0 : vector<1x1x1xi1> to vector<1xi1>
  return %1 : vector<1xi1>
}

//   CHECK-LABEL: func.func @create_mask
//    CHECK-SAME:     %[[IDX:.+]]: index
//     CHECK-DAG:   %[[TRUE:.+]] = arith.constant dense<true> : vector<1xi1>
//     CHECK-DAG:   %[[FALSE:.+]] = arith.constant dense<false> : vector<1xi1>
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//         CHECK:   %[[CMP:.+]] = arith.cmpi sgt, %[[IDX]], %[[C0]] : index
//         CHECK:   %[[MASK:.+]] = arith.select %[[CMP]], %[[TRUE]], %[[FALSE]] : vector<1xi1>
//         CHECK:   return %[[MASK]] : vector<1xi1>
