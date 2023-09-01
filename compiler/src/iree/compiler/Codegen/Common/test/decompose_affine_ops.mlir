// RUN: iree-opt -decompose-affine-ops %s | FileCheck %s

// Check that we have one affine.apply per loop dependence:
// IV0 with STRIDE0, IV1 with STRID1, etc.
// And that we combine them together following the nesting level
// of the loop: IV2 -> IV1 -> IV0
// CHECK: #[[$SINGLE_VALUE_MAP:.*]] = affine_map<()[s0] -> (s0)>
// CHECK: #[[$MUL_MAP:.*]] = affine_map<()[s0, s1] -> (s1 * s0)>
// CHECK: #[[$ADD_MAP:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL:   func.func @decomposeAffine(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> f32 {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-DAG:       %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-DAG:       %[[VAL_7:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[VAL_0]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
// CHECK:           %[[VAL_12:.*]] = scf.for %[[IV2:.*]] = %[[VAL_2]] to %[[VAL_7]] step %[[VAL_3]] iter_args(%[[VAL_14:.*]] = %[[VAL_1]]) -> (f32) {
// CHECK:             %[[VAL_15:.*]] = scf.for %[[IV1:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_3]] iter_args(%[[VAL_17:.*]] = %[[VAL_14]]) -> (f32) {
// CHECK:               %[[VAL_18:.*]] = scf.for %[[IV0:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_3]] iter_args(%[[VAL_20:.*]] = %[[VAL_17]]) -> (f32) {
// CHECK:                 %[[VAL_21:.*]] = affine.apply #[[$SINGLE_VALUE_MAP]](){{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_22:.*]] = affine.apply #[[$MUL_MAP]](){{\[}}%[[STRIDES]]#2, %[[IV2]]]
// CHECK:                 %[[VAL_23:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[VAL_21]], %[[VAL_22]]]
// CHECK:                 %[[VAL_24:.*]] = affine.apply #[[$MUL_MAP]](){{\[}}%[[STRIDES]]#1, %[[IV1]]]
// CHECK:                 %[[VAL_25:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[VAL_23]], %[[VAL_24]]]
// CHECK:                 %[[VAL_26:.*]] = affine.apply #[[$MUL_MAP]](){{\[}}%[[STRIDES]]#0, %[[IV0]]]
// CHECK:                 %[[VAL_27:.*]] = affine.apply #[[$ADD_MAP]](){{\[}}%[[VAL_25]], %[[VAL_26]]]
// CHECK:                 %[[VAL_28:.*]] = memref.reinterpret_cast %[[VAL_8]] to offset: {{\[}}%[[VAL_27]]], sizes: [1, 1, 1], strides: {{\[}}%[[STRIDES]]#0, %[[STRIDES]]#1, %[[STRIDES]]#2] : memref<f32> to memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:                 %[[VAL_29:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_2]], %[[VAL_2]], %[[VAL_2]]] : memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:                 %[[VAL_30:.*]] = arith.addf %[[VAL_29]], %[[VAL_14]] : f32
// CHECK:                 scf.yield %[[VAL_30]] : f32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_31:.*]] : f32
// CHECK:             }
// CHECK:             scf.yield %[[VAL_32:.*]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_33:.*]] : f32
// CHECK:         }
func.func @decomposeAffine(%arg0: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> f32 {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim = memref.dim %arg0, %c0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %dim_1 = memref.dim %arg0, %c2 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
  %0 = scf.for %arg1 = %c0 to %dim_1 step %c1 iter_args(%arg2 = %cst) -> (f32) {
    %1 = scf.for %arg3 = %c0 to %dim_0 step %c1 iter_args(%arg4 = %arg2) -> (f32) {
      %2 = scf.for %arg5 = %c0 to %dim step %c1 iter_args(%arg6 = %arg4) -> (f32) {
        %3 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 + s1 * s2 + s3 * s4 + s5 * s6)>()[%offset, %arg5, %strides#0, %arg3, %strides#1, %arg1, %strides#2]
        %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%3], sizes: [1, 1, 1], strides: [%strides#0, %strides#1, %strides#2] : memref<f32> to memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
        %4 = memref.load %reinterpret_cast[%c0, %c0, %c0] : memref<1x1x1xf32, strided<[?, ?, ?], offset: ?>>
        %5 = arith.addf %4, %arg2 : f32
        scf.yield %5 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  }
  return %0 : f32
}
