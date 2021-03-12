// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions2 -iree-enable-consumer-only-fusion -canonicalize %s | IreeFileCheck %s

func @simpleDotAddMul
  (%arg0 : tensor<16x32xf32>, %arg1 : tensor<32x48xf32>,
   %arg2 : tensor<16x48xf32>, %arg3 : tensor<16x48xf32>) -> tensor<16x48xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) :
    (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
  %1 = mhlo.add %0, %arg2 : tensor<16x48xf32>
  %2 = mhlo.multiply %1, %arg3 : tensor<16x48xf32>
  return %2 : tensor<16x48xf32>
}
// CHECK-LABEL: func @simpleDotAddMul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x32xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x48xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<16x48xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<16x48xf32>
//  CHECK-NEXT:   %[[WORKLOAD:.+]] = constant 768
//  CHECK-NEXT:   %[[RESULT:.+]] = flow.dispatch.region[%[[WORKLOAD]] : index]
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG0]]
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] = %[[ARG1]]
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG2]]
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG3]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T1:.+]] = "mhlo.dot"(%[[ARG4]], %[[ARG5]])
//  CHECK-NEXT:       %[[T2:.+]] = mhlo.add %[[T1]], %[[ARG6]]
//  CHECK-NEXT:       %[[T3:.+]] = mhlo.multiply %[[T2]], %[[ARG7]]
//  CHECK-NEXT:       flow.return %[[T3]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   return %[[RESULT]]

// -----

func @twoDots
  (%arg0 : tensor<16x32xf32>, %arg1 : tensor<32x48xf32>,
   %arg2 : tensor<16x48xf32>, %arg3 : tensor<16x64xf32>,
   %arg4 : tensor<16x64xf32>) -> tensor<16x64xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) :
    (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
  %1 = mhlo.add %0, %arg2 : tensor<16x48xf32>
  %2 = "mhlo.dot"(%1, %arg3) :
    (tensor<16x48xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
  %3 = mhlo.multiply %2, %arg4 : tensor<16x64xf32>
  return %3 : tensor<16x64xf32>
}
// CHECK-LABEL: func @twoDots
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<16x32xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<32x48xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<16x48xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<16x64xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: tensor<16x64xf32>
//  CHECK-NEXT:   %[[WORKLOAD1:.+]] = constant 1024
//  CHECK-NEXT:   %[[WORKLOAD2:.+]] = constant 768
//  CHECK-NEXT:   %[[RESULT1:.+]] = flow.dispatch.region[%[[WORKLOAD2]] : index]
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] = %[[ARG0]]
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG1]]
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG2]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T1:.+]] = "mhlo.dot"(%[[ARG5]], %[[ARG6]])
//  CHECK-NEXT:       %[[T2:.+]] = mhlo.add %[[T1]], %[[ARG7]]
//  CHECK-NEXT:       flow.return %[[T2]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   %[[RESULT2:.+]] = flow.dispatch.region[%[[WORKLOAD1]] : index]
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] = %[[RESULT1]]
//  CHECK-SAME:     %[[ARG6:[a-zA-Z0-9_]+]] = %[[ARG3]]
//  CHECK-SAME:     %[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG4]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T3:.+]] = "mhlo.dot"(%[[ARG5]], %[[ARG6]])
//  CHECK-NEXT:       %[[T4:.+]] = mhlo.multiply %[[T3]], %[[ARG7]]
//  CHECK-NEXT:       flow.return %[[T4]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   return %[[RESULT2]]

// -----

func @moveDispatchOp
  (%arg0 : tensor<1x384x384xf32>, %arg1 : tensor<384x512xf32>,
   %arg2 : tensor<512xf32>) -> tensor<1x384x512xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x384x384xf32>) -> tensor<384x384xf32>
  %1 = "mhlo.dot"(%0, %arg1) :
    (tensor<384x384xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>
  %2 = "mhlo.broadcast_in_dim"(%arg2)
    {broadcast_dimensions = dense<1> : tensor<1xi64>} :
    (tensor<512xf32>) -> tensor<384x512xf32>
  %3 = mhlo.add %1, %2 : tensor<384x512xf32>
  %4 = "mhlo.reshape"(%3) : (tensor<384x512xf32>) -> tensor<1x384x512xf32>
  return %4 : tensor<1x384x512xf32>
}
// CHECK-LABEL: func @moveDispatchOp
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x384x384xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<384x512xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<512xf32>
//       CHECK:   %[[RESULT1:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[ARG2]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T1:.+]] = "mhlo.broadcast_in_dim"(%[[ARG3]])
//  CHECK-NEXT:       flow.return %[[T1]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   %[[RESULT2:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[ARG1]]
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]] = %[[RESULT1]]
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] = %[[ARG0]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T2:.+]] = "mhlo.reshape"(%[[ARG5]])
//  CHECK-NEXT:       %[[T3:.+]] = "mhlo.dot"(%[[T2]], %[[ARG3]])
//  CHECK-NEXT:       %[[T4:.+]] = mhlo.add %[[T3]], %[[ARG4]]
//  CHECK-NEXT:       %[[T5:.+]] = "mhlo.reshape"(%[[T4]])
//  CHECK-NEXT:       flow.return %[[T5]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   return %[[RESULT2]]

// -----

func @dot_fusion_with_different_shape
  (%arg0: tensor<384x512xf32>, %arg1: tensor<512x128xf32>,
   %arg2: tensor<384x128xf32>) -> tensor<4x384x32xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1)
    : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
  %1 = mhlo.add %0, %arg2 : tensor<384x128xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<384x128xf32>) -> tensor<1x384x4x32xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>}
    : (tensor<1x384x4x32xf32>) -> tensor<1x4x384x32xf32>
  %4 = "mhlo.reshape"(%3) : (tensor<1x4x384x32xf32>) -> tensor<4x384x32xf32>
  return %4 : tensor<4x384x32xf32>
}

// CHECK-LABEL: func @dot_fusion_with_different_shape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<384x512xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<512x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<384x128xf32>
//       CHECK:   %[[RESULT1:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[ARG0]]
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG1]]
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]] = %[[ARG2]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T2:.+]] = "mhlo.dot"(%[[ARG3]], %[[ARG4]])
//  CHECK-NEXT:       %[[T3:.+]] = mhlo.add %[[T2]], %[[ARG5]]
//  CHECK-NEXT:       %[[T4:.+]] = "mhlo.reshape"(%[[T3]])
//  CHECK-NEXT:       flow.return %[[T4]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   %[[RESULT2:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[RESULT1]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T2:.+]] = "mhlo.transpose"(%[[ARG3]])
//  CHECK-NEXT:       %[[T3:.+]] = "mhlo.reshape"(%[[T2]])
//  CHECK-NEXT:       flow.return %[[T3]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   return %[[RESULT2]]

// -----

func @dot_general_lower_swapped
  (%arg0 : tensor<2x3xf32>, %arg1 : tensor<1x1x2xf32>) -> tensor<3x1x1xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>}
    : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %1 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 0, 1]> : tensor<3xi64>}
    : (tensor<1x1x2xf32>) -> tensor<2x1x1xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
  %3 = "mhlo.dot"(%0, %2) {precision_config = ["DEFAULT", "DEFAULT"]}
    : (tensor<3x2xf32>, tensor<2x1xf32>) -> tensor<3x1xf32>
  %4 = "mhlo.reshape"(%3) : (tensor<3x1xf32>) -> tensor<3x1x1xf32>
  return %4 : tensor<3x1x1xf32>
}
// CHECK-LABEL: func @dot_general_lower_swapped
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x3xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x2xf32>
//       CHECK:   %[[RESULT1:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]] = %[[ARG0]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T3:.+]] = "mhlo.transpose"(%[[ARG2]])
//  CHECK-NEXT:       flow.return %[[T3]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   %[[RESULT2:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]] = %[[ARG1]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T3:.+]] = "mhlo.transpose"(%[[ARG2]])
//  CHECK-NEXT:       flow.return %[[T3]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   %[[RESULT3:.+]] = flow.dispatch.region
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]] = %[[RESULT1]]
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[RESULT2]]
//  CHECK-SAME:     {
//  CHECK-NEXT:       %[[T3:.+]] = "mhlo.reshape"(%[[ARG3]])
//  CHECK-NEXT:       %[[T4:.+]] = "mhlo.dot"(%[[ARG2]], %[[T3]])
//  CHECK-NEXT:       %[[T5:.+]] = "mhlo.reshape"(%[[T4]])
//  CHECK-NEXT:       flow.return %[[T5]]
//  CHECK-NEXT:     }
//  CHECK-NEXT:   return %[[RESULT3]]
