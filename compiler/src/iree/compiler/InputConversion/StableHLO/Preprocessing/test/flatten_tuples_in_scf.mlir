// RUN: iree-opt --iree-stablehlo-preprocessing-flatten-scf-tuples %s | FileCheck %s

func.func @testWhile(%arg0 : i32, %arg1 : tuple<tensor<4xf32>, tensor<4xf32>>) -> (i32, tuple<tensor<4xf32>, tensor<4xf32>>) {
  %0:2 = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (i32, tuple<tensor<4xf32>, tensor<4xf32>>) -> (i32, tuple<tensor<4xf32>, tensor<4xf32>>) {
    %c10 = arith.constant 10 : i32
    %1 = arith.cmpi slt, %arg2, %c10 : i32
    scf.condition(%1) %arg2, %arg3 : i32, tuple<tensor<4xf32>, tensor<4xf32>>
  } do {
  ^bb0(%arg2: i32, %arg3: tuple<tensor<4xf32>, tensor<4xf32>>):
    %c1 = arith.constant 1 : i32
    %add = arith.addi %arg2, %c1 : i32
    %2 = stablehlo.get_tuple_element %arg3[0] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    %3 = stablehlo.get_tuple_element %arg3[1] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    %4 = stablehlo.add %2, %3 : tensor<4xf32>
    %5 = stablehlo.tuple %3, %4 : tuple<tensor<4xf32>, tensor<4xf32>>
    scf.yield %add, %5 : i32, tuple<tensor<4xf32>, tensor<4xf32>>
  }
  return %0#0, %0#1 : i32, tuple<tensor<4xf32>, tensor<4xf32>>
}

// CHECK-LABEL: @testWhile
// CHECK-SAME: %[[ARG0:.+]]: i32, %[[ARG1:.+]]: tuple<tensor<4xf32>, tensor<4xf32>>
// CHECK: %[[C1:.+]] = arith.constant 1
// CHECK: %[[C10:.+]] = arith.constant 10
// CHECK: %[[L:.+]] = stablehlo.get_tuple_element %[[ARG1]][0]
// CHECK: %[[R:.+]] = stablehlo.get_tuple_element %[[ARG1]][1]

// CHECK: %[[WHILE:.+]]:3 = scf.while (%[[ARG2:.+]] = %[[ARG0]], %[[ARG3:.+]] = %[[L]], %[[ARG4:.+]] = %[[R]])
// CHECK:   %[[CMP:.+]] = arith.cmpi slt, %[[ARG2]], %[[C10]]
// CHECK:   scf.condition(%[[CMP]]) %[[ARG2]], %[[ARG3]], %[[ARG4]]

// CHECK: ^bb0(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: tensor<4xf32>, %[[ARG4:.+]]: tensor<4xf32>):
// CHECK:   %[[ADD:.+]] = arith.addi %[[ARG2]], %[[C1]] : i32
// CHECK:   %[[SADD:.+]] = stablehlo.add %[[ARG3]], %[[ARG4]] : tensor<4xf32>
// CHECK:   scf.yield %[[ADD]], %[[ARG4]], %[[SADD]]

// CHECK: %[[TUPLE:.+]] = stablehlo.tuple %[[WHILE]]#1, %[[WHILE]]#2 : tuple<tensor<4xf32>, tensor<4xf32>>
// CHECK: return %[[WHILE]]#0, %[[TUPLE]]

// -----

func.func @testIf(%cond: i1, %a : tuple<tensor<4xf32>, tensor<4xf32>>) -> (tuple<tensor<4xf32>, tensor<4xf32>>) {
  %r = scf.if %cond -> (tuple<tensor<4xf32>, tensor<4xf32>>) {
    %2 = stablehlo.get_tuple_element %a[0] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    %3 = stablehlo.get_tuple_element %a[1] : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
    %5 = stablehlo.tuple %3, %2 : tuple<tensor<4xf32>, tensor<4xf32>>
    scf.yield %5 : tuple<tensor<4xf32>, tensor<4xf32>>
  } else {
    scf.yield %a : tuple<tensor<4xf32>, tensor<4xf32>>
  }
  return %r : tuple<tensor<4xf32>, tensor<4xf32>>
}

// CHECK-LABEL: @testIf
// CHECK-SAME: %[[ARG0:.+]]: i1, %[[ARG1:.+]]: tuple<tensor<4xf32>, tensor<4xf32>>

// CHECK:  %[[IF:.+]]:2 = scf.if %[[ARG0]] -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK:    %[[L:.+]] = stablehlo.get_tuple_element %[[ARG1]][0]
// CHECK:    %[[R:.+]] = stablehlo.get_tuple_element %[[ARG1]][1]
// CHECK:    scf.yield %[[R]], %[[L]]

// CHECK:    %[[L:.+]] = stablehlo.get_tuple_element %[[ARG1]][0]
// CHECK:    %[[R:.+]] = stablehlo.get_tuple_element %[[ARG1]][1]
// CHECK:    scf.yield %[[L]], %[[R]] : tensor<4xf32>, tensor<4xf32>
// CHECK:  %[[TUPLE:.+]] = stablehlo.tuple %[[IF]]#0, %[[IF]]#1 : tuple<tensor<4xf32>, tensor<4xf32>>
// CHECK:  return %[[TUPLE]]
