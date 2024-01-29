// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-vectorize-tensor-pad),canonicalize,cse)" %s | FileCheck %s

// Note: A pull request is open to upstream this pattern:
//   https://reviews.llvm.org/D117021
// Once it lands, this pattern can be replaced.

func.func @tensor_pad(%source: tensor<1x?x?x3xf32>, %low1: index, %low2: index, %high1: index, %high2: index) -> tensor<1x2x2x3xf32> {
  %cst = arith.constant 0.0 : f32
  %pad = tensor.pad %source low[0, %low1, %low2, 0] high[0, %high1, %high2, 0]  {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x?x?x3xf32> to tensor<1x2x2x3xf32>
  return %pad: tensor<1x2x2x3xf32>
}

// CHECK-LABEL: func.func @tensor_pad
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<1x?x?x3xf32>, %[[LOW1:.+]]: index, %[[LOW2:.+]]: index, %{{.+}}: index, %{{.+}}: index)

// CHECK-DAG:   %[[I0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[V3F0:.+]] = arith.constant dense<0.000000e+00> : vector<3xf32>
// CHECK-DAG:   %[[FULL:.+]] = arith.constant dense<0.000000e+00> : vector<2x2x3xf32>
// CHECK-DAG:   %[[I2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[I1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

// CHECK:   %[[DIM1:.+]] = tensor.dim %[[SOURCE]], %[[I1]]
// CHECK:   %[[UB1:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[LOW1]], %[[DIM1]]]
// CHECK:   %[[DIM2:.+]] = tensor.dim %[[SOURCE]], %[[I2]]
// CHECK:   %[[UB2:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[LOW2]], %[[DIM2]]]

// CHECK:   %[[LE:.+]] = arith.cmpi sle, %[[LOW1]], %[[I0]]
// CHECK:   %[[GT:.+]] = arith.cmpi sgt, %[[UB1]], %[[I0]]
// CHECK:   %[[DIM1INDEX0INBOUND:.+]] = arith.andi %[[LE]], %[[GT]]
// CHECK:   %[[LE:.+]] = arith.cmpi sle, %[[LOW2]], %[[I0]]
// CHECK:   %[[GT:.+]] = arith.cmpi sgt, %[[UB2]], %[[I0]]
// CHECK:   %[[DIM2INDEX0INBOUND:.+]] = arith.andi %[[LE]], %[[GT]]
// CHECK:   %[[AND0:.+]] = arith.andi %[[DIM1INDEX0INBOUND]], %[[DIM2INDEX0INBOUND]]
// CHECK:   %[[DIM1INDEX0:.+]] = affine.apply affine_map<()[s0] -> (-s0)>()[%[[LOW1]]]
// CHECK:   %[[DIM2INDEX0:.+]] = affine.apply affine_map<()[s0] -> (-s0)>()[%[[LOW2]]]
// CHECK:   %[[IF0:.+]] = scf.if %[[AND0]] -> (vector<3xf32>) {
// CHECK:     %[[READ:.+]] = vector.transfer_read %[[SOURCE]][%[[I0]], %[[DIM1INDEX0]], %[[DIM2INDEX0]], %[[I0]]], %[[F0]] {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
// CHECK:     scf.yield %[[READ]] : vector<3xf32>
// CHECK:   } else {
// CHECK:     scf.yield %[[V3F0]] : vector<3xf32>
// CHECK:   }
// CHECK:   %[[INSERT0:.+]] = vector.insert_strided_slice %[[IF0]], %[[FULL]] {offsets = [0, 0, 0], strides = [1]} : vector<3xf32> into vector<2x2x3xf32>

// CHECK:   %[[LE:.+]] = arith.cmpi sle, %[[LOW2]], %[[I1]]
// CHECK:   %[[GT:.+]] = arith.cmpi sgt, %[[UB2]], %[[I1]]
// CHECK:   %[[DIM2INDEX1INBOUND:.+]] = arith.andi %[[LE]], %[[GT]]
// CHECK:   %[[AND1:.+]] = arith.andi %[[DIM1INDEX0INBOUND]], %[[DIM2INDEX1INBOUND]]
// CHECK:   %[[DIM2INDEX1:.+]] = affine.apply affine_map<()[s0] -> (-s0 + 1)>()[%[[LOW2]]]
// CHECK:   %[[IF1:.+]] = scf.if %[[AND1]] -> (vector<3xf32>) {
// CHECK:     %[[READ:.+]] = vector.transfer_read %[[SOURCE]][%[[I0]], %[[DIM1INDEX0]], %[[DIM2INDEX1]], %[[I0]]], %[[F0]] {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
// CHECK:     scf.yield %[[READ]] : vector<3xf32>
// CHECK:   } else {
// CHECK:     scf.yield %[[V3F0]] : vector<3xf32>
// CHECK:   }
// CHECK:   %[[INSERT1:.+]] = vector.insert_strided_slice %[[IF1]], %[[INSERT0]] {offsets = [0, 1, 0], strides = [1]} : vector<3xf32> into vector<2x2x3xf32>

// CHECK:   %[[LE:.+]] = arith.cmpi sle, %[[LOW1]], %[[I1]]
// CHECK:   %[[GT:.+]] = arith.cmpi sgt, %[[UB1]], %[[I1]]
// CHECK:   %[[DIM1INDEX1INBOUND:.+]] = arith.andi %[[LE]], %[[GT]]
// CHECK:   %[[AND2:.+]] = arith.andi %[[DIM1INDEX1INBOUND]], %[[DIM2INDEX0INBOUND]]
// CHECK:   %[[DIM1INDEX1:.+]] = affine.apply affine_map<()[s0] -> (-s0 + 1)>()[%[[LOW1]]]
// CHECK:   %[[IF2:.+]] = scf.if %[[AND2]] -> (vector<3xf32>) {
// CHECK:     %[[READ:.+]] = vector.transfer_read %[[SOURCE]][%[[I0]], %[[DIM1INDEX1]], %[[DIM2INDEX0]], %[[I0]]], %[[F0]] {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
// CHECK:     scf.yield %[[READ]] : vector<3xf32>
// CHECK:   } else {
// CHECK:     scf.yield %[[V3F0]] : vector<3xf32>
// CHECK:   }
// CHECK:   %[[INSERT2:.+]] = vector.insert_strided_slice %[[IF2]], %[[INSERT1]] {offsets = [1, 0, 0], strides = [1]} : vector<3xf32> into vector<2x2x3xf32>

// CHECK:   %[[AND3:.+]] = arith.andi %[[DIM1INDEX1INBOUND]], %[[DIM2INDEX1INBOUND]]
// CHECK:   %[[IF3:.+]] = scf.if %[[AND3]] -> (vector<3xf32>) {
// CHECK:     %[[READ:.+]] = vector.transfer_read %[[SOURCE]][%[[I0]], %[[DIM1INDEX1]], %[[DIM2INDEX1]], %[[I0]]], %[[F0]] {in_bounds = [true]} : tensor<1x?x?x3xf32>, vector<3xf32>
// CHECK:     scf.yield %[[READ]] : vector<3xf32>
// CHECK:   } else {
// CHECK:     scf.yield %[[V3F0]] : vector<3xf32>
// CHECK:   }
// CHECK:   %[[INSERT3:.+]] = vector.insert_strided_slice %[[IF3]], %[[INSERT2]] {offsets = [1, 1, 0], strides = [1]} : vector<3xf32> into vector<2x2x3xf32>

// CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x2x2x3xf32>
// CHECK:   %[[WRITE:.+]] = vector.transfer_write %[[INSERT3]], %[[INIT]][%[[I0]], %[[I0]], %[[I0]], %[[I0]]] {in_bounds = [true, true, true]} : vector<2x2x3xf32>, tensor<1x2x2x3xf32>
// CHECK:   return %[[WRITE]]

// -----

func.func @no_vectorize_tensor_pad(%source: tensor<1x?x?x3xf32>, %low1: index,
    %low2: index, %high1: index, %high2: index, %arg0 : f32) -> tensor<1x2x2x3xf32> {
  %pad = tensor.pad %source low[0, %low1, %low2, 0] high[0, %high1, %high2, 0]  {
  ^bb0(%b0: index, %b1: index, %b2: index, %b3: index):
    tensor.yield %arg0 : f32
  } : tensor<1x?x?x3xf32> to tensor<1x2x2x3xf32>
  return %pad: tensor<1x2x2x3xf32>
}
// CHECK-LABEL: func @no_vectorize_tensor_pad
//       CHECK:   tensor.pad
