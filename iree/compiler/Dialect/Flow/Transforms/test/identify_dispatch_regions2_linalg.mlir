// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions2 %s | IreeFileCheck %s

func @constant_capture(%arg0 : tensor<10x20xf32>) -> tensor<10x20xf32> {
  %cst1 = constant 1.0 : f32
  %cst2 = constant dense<2.0> : tensor<10x20xf32>
  %cst3 = constant dense<
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>
  %0 = linalg.init_tensor [10, 20] : tensor<10x20xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0, d1)>,
                      affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %cst2, %cst3
      : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10xf32>)
    outs(%0 : tensor<10x20xf32>) {
    ^bb0(%arg1 : f32, %arg2 : f32, %arg3 : f32, %arg4 : f32):
      %1 = addf %arg1, %cst1 : f32
      %2 = mulf %1, %arg2 : f32
      %3 = addf %2, %arg3 : f32
      linalg.yield %3 : f32
    } -> tensor<10x20xf32>
  return %1 : tensor<10x20xf32>
}
//       CHECK: func @constant_capture
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<10x20xf32>
//   CHECK-DAG:   %[[CST1:.+]] = constant 1.000000e+00 : f32
//   CHECK-DAG:   %[[CST2:.+]] = constant dense<2.000000e+00> : tensor<10x20xf32>
//   CHECK-DAG:   %[[CST3:.+]] = constant dense<[1.000000e+00, 2.000000e+00,
//  CHECK-SAME:     3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00,
//  CHECK-SAME:     7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]>
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.region[%{{.+}} : index](
//  CHECK-SAME:       %[[ARG1:[a-zA-Z0-9_]+]] = %[[ARG0]]
//  CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]] = %[[CST2]]
//  CHECK-SAME:       %[[ARG3:[a-zA-Z0-9_]+]] = %[[CST3]]
//  CHECK-SAME:       %[[ARG4:[a-zA-Z0-9_]+]] = %[[CST1]]
//  CHECK-SAME:     ) -> tensor<10x20xf32> {
//       CHECK:     %[[T0:.+]] = linalg.init_tensor [10, 20] : tensor<10x20xf32>
//       CHECK:     %[[RETURN:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]], %[[ARG2]], %[[ARG3]]
//  CHECK-SAME:       ) outs(%[[T0]] : tensor<10x20xf32>) {
//  CHECK-NEXT:       ^{{[a-zA-Z0-9]+}}(
//  CHECK-SAME:         %[[ARG5:.[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:         %[[ARG6:.[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:         %[[ARG7:.[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:         %[[ARG8:.[a-zA-Z0-9_]+]]: f32)
//       CHECK:         %[[T0:.+]] = addf %[[ARG5]], %[[ARG4]]
//       CHECK:         %[[T1:.+]] = mulf %[[T0]], %[[ARG6]]
//       CHECK:         %[[T2:.+]] = addf %[[T1]], %[[ARG7]]
//       CHECK:         linalg.yield %[[T2]]
//       CHECK:       }
//       CHECK:     flow.return %[[RETURN]]
//       CHECK:   }
