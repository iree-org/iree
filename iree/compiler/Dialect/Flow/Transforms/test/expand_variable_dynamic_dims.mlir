// RUN: iree-opt -split-input-file -iree-flow-expand-variable-dynamic-dims %s | IreeFileCheck %s

// CHECK-NOT: flow.variable @static_var mutable
// CHECK: flow.variable @static_var_d0 1 : index
// CHECK: flow.variable @static_var_d1 2 : index
// CHECK: flow.variable @static_var_d2 3 : index
// CHECK: flow.variable @static_var_d3 4 : index
flow.variable @static_var mutable : !shapex.ranked_shape<[1,2,3,4]>
// CHECK-LABEL: func @static_loads
func @static_loads() -> (index, index, index, index) {
  // CHECK-NOT: flow.variable.load
  // CHECK-NEXT: %[[SHAPE:.+]] = shapex.make_ranked_shape
  %0 = flow.variable.load @static_var : !shapex.ranked_shape<[1,2,3,4]>
  // CHECK-DAG: %[[D0:.+]] = shapex.ranked_dim %[[SHAPE]][0]
  %1 = shapex.ranked_dim %0[0] : !shapex.ranked_shape<[1,2,3,4]> -> index
  // CHECK-DAG: %[[D1:.+]] = shapex.ranked_dim %[[SHAPE]][1]
  %2 = shapex.ranked_dim %0[1] : !shapex.ranked_shape<[1,2,3,4]> -> index
  // CHECK-DAG: %[[D2:.+]] = shapex.ranked_dim %[[SHAPE]][2]
  %3 = shapex.ranked_dim %0[2] : !shapex.ranked_shape<[1,2,3,4]> -> index
  // CHECK-DAG: %[[D3:.+]] = shapex.ranked_dim %[[SHAPE]][3]
  %4 = shapex.ranked_dim %0[3] : !shapex.ranked_shape<[1,2,3,4]> -> index
  // CHECK-NEXT: return %[[D0]], %[[D1]], %[[D2]], %[[D3]]
  return %1, %2, %3, %4 : index, index, index, index
}
// CHECK-LABEL: func @static_stores
func @static_stores(%arg0 : index, %arg1 : index) {
  // CHECK-NEXT: %[[SHAPE:.+]] = shapex.const_ranked_shape
  %0 = shapex.const_ranked_shape : !shapex.ranked_shape<[1,2,3,4]>
  // CHECK-NOT: flow.variable.store
  flow.variable.store %0, @static_var : !shapex.ranked_shape<[1,2,3,4]>
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-NOT: flow.variable @dynamic_var mutable
// CHECK: flow.variable @dynamic_var_d0 1 : index
// CHECK: flow.variable @dynamic_var_d1 mutable 0 : index
// CHECK: flow.variable @dynamic_var_d2 mutable 0 : index
// CHECK: flow.variable @dynamic_var_d3 4 : index
flow.variable @dynamic_var mutable : !shapex.ranked_shape<[1,?,?,4]>
// CHECK-LABEL: func @dynamic_loads
func @dynamic_loads() -> (index, index, index, index) {
  // CHECK-NOT: flow.variable.load @dynamic_var
  // CHECK-DAG: %[[D1:.+]] = flow.variable.load @dynamic_var_d1 : index
  // CHECK-DAG: %[[D2:.+]] = flow.variable.load @dynamic_var_d2 : index
  // CHECK-NEXT: %[[SHAPE:.+]] = shapex.make_ranked_shape %[[D1]], %[[D2]] : (index, index) -> !shapex.ranked_shape<[1,?,?,4]>
  %0 = flow.variable.load @dynamic_var : !shapex.ranked_shape<[1,?,?,4]>
  // CHECK-DAG: = shapex.ranked_dim %[[SHAPE]][0] : !shapex.ranked_shape<[1,?,?,4]> -> index
  %1 = shapex.ranked_dim %0[0] : !shapex.ranked_shape<[1,?,?,4]> -> index
  // CHECK-DAG: = shapex.ranked_dim %[[SHAPE]][1] : !shapex.ranked_shape<[1,?,?,4]> -> index
  %2 = shapex.ranked_dim %0[1] : !shapex.ranked_shape<[1,?,?,4]> -> index
  // CHECK-DAG: = shapex.ranked_dim %[[SHAPE]][2] : !shapex.ranked_shape<[1,?,?,4]> -> index
  %3 = shapex.ranked_dim %0[2] : !shapex.ranked_shape<[1,?,?,4]> -> index
  // CHECK-DAG: = shapex.ranked_dim %[[SHAPE]][3] : !shapex.ranked_shape<[1,?,?,4]> -> index
  %4 = shapex.ranked_dim %0[3] : !shapex.ranked_shape<[1,?,?,4]> -> index
  // CHECK-NEXT: return
  return %1, %2, %3, %4 : index, index, index, index
}
// CHECK-LABEL: func @dynamic_stores
// CHECK-SAME: (%[[D1:.+]]: index, %[[D2:.+]]: index)
func @dynamic_stores(%arg0 : index, %arg1 : index) {
  // CHECK-NEXT: %[[SHAPE:.+]] = shapex.make_ranked_shape %arg0, %arg1
  %0 = shapex.make_ranked_shape %arg0, %arg1 : (index, index) -> !shapex.ranked_shape<[1,?,?,4]>
  // CHECK-NEXT: %[[D1:.+]] = shapex.ranked_dim %[[SHAPE]][1] : !shapex.ranked_shape<[1,?,?,4]> -> index
  // CHECK-NEXT: flow.variable.store %[[D1]], @dynamic_var_d1 : index
  // CHECK-NEXT: %[[D2:.+]] = shapex.ranked_dim %[[SHAPE]][2] : !shapex.ranked_shape<[1,?,?,4]> -> index
  // CHECK-NEXT: flow.variable.store %[[D2]], @dynamic_var_d2 : index
  flow.variable.store %0, @dynamic_var : !shapex.ranked_shape<[1,?,?,4]>
  // CHECK-NEXT: return
  return
}
