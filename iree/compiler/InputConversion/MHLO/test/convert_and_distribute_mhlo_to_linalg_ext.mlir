// RUN: iree-opt -split-input-file -iree-mhlo-to-linalg-ext %s | IreeFileCheck %s

func @sort_1d(%arg0: tensor<128xi32>) -> (tensor<128xi32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xi32>) -> (tensor<128xi32>)
  return %0 : tensor<128xi32>
}
// CHECK-LABEL: func @sort_1d
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]]) : (tensor<128xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG1:.+]]: !flow.dispatch.tensor<readwrite:128xi32>
// CHECK:           %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]]
// CHECK:           %[[SORT:.+]] = linalg_ext.sort
// CHECK-SAME:        dimension = 0 : i64
// CHECK-SAME:        outs(%[[IN]] : tensor<128xi32>)
// CHECK:          ^bb0(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: i32)
// CHECK:            %[[CMP:.+]] = cmpi sgt, %[[ARG2]], %[[ARG3]]
// CHECK:            linalg_ext.yield %[[CMP]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]], %[[ARG1]]
// CHECK:        return %[[RES]]

// -----

func @sort_2d(%arg0: tensor<16x32xi32>) -> (tensor<16x32xi32>) {
  %0 = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<16x32xi32>) -> (tensor<16x32xi32>)
  return %0 : tensor<16x32xi32>
}
// CHECK-LABEL: func @sort_2d
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]] = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]]) : (tensor<16x32xi32>) -> %[[ARG0]]
// CHECK:           %[[ARG1:.+]]: !flow.dispatch.tensor<readwrite:16x32xi32>
// CHECK:           %[[IN:.+]] = flow.dispatch.tensor.load %[[ARG1]]
// CHECK:           %[[SORT:.+]] = linalg_ext.sort
// CHECK-SAME:        dimension = 0 : i64
// CHECK-SAME:        outs(%[[IN]] : tensor<16x32xi32>)
// CHECK:          ^bb0(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: i32)
// CHECK:            %[[CMP:.+]] = cmpi sgt, %[[ARG2]], %[[ARG3]]
// CHECK:            linalg_ext.yield %[[CMP]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]], %[[ARG1]]
// CHECK:        return %[[RES]]

// -----

func @topk(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> (tensor<128xi32>) {
  %0:2 = "mhlo.sort"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):  // no predecessors
    %1 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<128xi32>, tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  return %0#0 : tensor<128xi32>
}
// CHECK-LABEL: func @topk
// CHECK:         %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1:.+]] = constant 1 : index
// CHECK:         %[[RES:.+]]:2 = flow.dispatch.workgroups
// CHECK-SAME:      [%[[C1]], %[[C1]], %[[C1]]](%[[ARG0]], %[[ARG1]])
// CHECK-SAME:    : (tensor<128xi32>, tensor<128xi32>) -> (%[[ARG0]], %[[ARG1]])
// CHECK:           %[[ARG2:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:128xi32>
// CHECK:           %[[ARG3:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readwrite:128xi32>
// CHECK:           %[[IN1:.+]] = flow.dispatch.tensor.load %[[ARG2]]
// CHECK:           %[[IN2:.+]] = flow.dispatch.tensor.load %[[ARG3]]
// CHECK:           %[[SORT:.+]]:2 = linalg_ext.sort
// CHECK-SAME:        dimension = 0 : i64
// CHECK-SAME:        outs(%[[IN1]], %[[IN2]] : tensor<128xi32>, tensor<128xi32>)
// CHECK:          ^bb0(%[[ARG4:.+]]: i32, %[[ARG5:.+]]: i32, %{{.*}}: i32, %{{.*}}: i32)
// CHECK:            %[[CMP:.+]] = cmpi sgt, %[[ARG4]], %[[ARG5]]
// CHECK:            linalg_ext.yield %[[CMP]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]]#0, %[[ARG2]]
// CHECK:          flow.dispatch.tensor.store %[[SORT]]#1, %[[ARG3]]
// CHECK:        return %[[RES]]#0
