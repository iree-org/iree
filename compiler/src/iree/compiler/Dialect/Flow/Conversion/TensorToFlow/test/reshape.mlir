// RUN: iree-opt --iree-flow-convert-to-flow --split-input-file %s | FileCheck %s

 util.func public @turn_fill_into_splat(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%1)[%arg2, %arg4]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%2)[%arg3, %arg5]
  %5 = tensor.empty(%3, %4) : tensor<?x?xf32>
  %6 = linalg.fill ins(%0 : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = flow.tensor.update %arg0, %6[%arg2, %arg3] : tensor<?x?xf32>{%1, %2} -> %6 as tensor<?x?xf32>{%3, %4}
  util.return %7 : tensor<?x?xf32>
}

//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>
//       CHECK:  util.func public @turn_fill_into_splat
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[VAL:.+]] = flow.tensor.load %[[ARG1]] : tensor<f32>
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[RD0:.+]] = affine.apply #[[MAP]](%[[D0]])[%[[ARG2]], %[[ARG4]]]
//   CHECK-DAG:   %[[RD1:.+]] = affine.apply #[[MAP]](%[[D1]])[%[[ARG3]], %[[ARG5]]]
//       CHECK:   %[[SPLAT:.+]] = flow.tensor.splat %[[VAL]] : tensor<?x?xf32>{%[[RD0]], %[[RD1]]}
//       CHECK:   flow.tensor.update %[[ARG0]], %[[SPLAT]]

// -----

 util.func public @static_tensor_reshape(%arg0: tensor<2x4xf32>, %arg1: tensor<2xindex>) -> tensor<1x8xf32> {
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<2x4xf32> -> tensor<1x8xf32>
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.reshape %arg0(%arg1)
             : (tensor<2x4xf32>, tensor<2xindex>) -> tensor<1x8xf32>
  util.return %0 : tensor<1x8xf32> }

// -----

   util.func public @dynamic_tensor_reshape(%arg0: tensor<2x4xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  //      CHECK:  util.func public @dynamic_tensor_reshape
  // CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4xf32>
  // CHECK-SAME:     %[[ARG1:.+]]: tensor<2xindex>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL:.+]] = flow.tensor.load %[[ARG1]][%[[C0]]] : tensor<2xindex>
  // CHECK-DAG: %[[VAL1:.+]] = flow.tensor.load %[[ARG1]][%[[C1]]] : tensor<2xindex>
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %[[ARG0]] : tensor<2x4xf32> -> tensor<?x?xf32>{%[[VAL]], %[[VAL1]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.reshape %arg0(%arg1)
             : (tensor<2x4xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32> }

  // -----

   util.func public @mix_dynamic_and_static_tensor_reshape(%arg0: tensor<2x4xf32>, %arg1: tensor<2xindex>) -> tensor<1x?xf32> {
  //      CHECK:  util.func public @mix_dynamic_and_static_tensor_reshape
  // CHECK-SAME:     %[[ARG0:.+]]: tensor<2x4xf32>
  // CHECK-SAME:     %[[ARG1:.+]]: tensor<2xindex>
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL:.+]] = flow.tensor.load %[[ARG1]][%[[C1]]] : tensor<2xindex>
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %[[ARG0]] : tensor<2x4xf32> -> tensor<1x?xf32>{%[[VAL]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.reshape %arg0(%arg1)
             : (tensor<2x4xf32>, tensor<2xindex>) -> tensor<1x?xf32>
  util.return %0 : tensor<1x?xf32> }

  // -----

   util.func public @dynamic_input_and_output_tensor_reshape(%arg0: tensor<?x4xf32>, %arg1: tensor<2xindex>) -> tensor<1x?xf32> {
  //      CHECK:  util.func public @dynamic_input_and_output_tensor_reshape
  // CHECK-SAME:     %[[ARG0:.+]]: tensor<?x4xf32>
  // CHECK-SAME:     %[[ARG1:.+]]: tensor<2xindex>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x4xf32>
  // CHECK-DAG: %[[VAL:.+]] = flow.tensor.load %[[ARG1]][%[[C1]]] : tensor<2xindex>
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %[[ARG0]] : tensor<?x4xf32>{%[[D0]]} -> tensor<1x?xf32>{%[[VAL]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.reshape %arg0(%arg1)
             : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<1x?xf32>
  util.return %0 : tensor<1x?xf32> }

  // -----
   util.func public @from_elements_test_reshape(%arg0: tensor<?x4xf32>, %arg1: index, %arg2: index) -> tensor<?x1xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[D1:.*]] = tensor.dim %arg0, %[[C0:.*]] : tensor<?x4xf32>
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.reshape %arg0 : tensor<?x4xf32>{%[[D1]]} -> tensor<?x1xf32>{%arg1}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.from_elements %arg1, %arg2 : tensor<2xindex>
  %1 = tensor.reshape %arg0(%0)
             : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x1xf32>
  util.return %1 : tensor<?x1xf32> }
