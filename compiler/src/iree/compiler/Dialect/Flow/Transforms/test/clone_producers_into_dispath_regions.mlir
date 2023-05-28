// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-flow-clone-producers-into-dispatch-regions))" %s | FileCheck %s

func.func @complex_element_type(%input: tensor<4xi32>, %table: tensor<8x2xcomplex<f32>>) -> tensor<4x2xcomplex<f32>> {
  %c4095 = arith.constant 4095 : i32
  %const = arith.constant dense<[
    [(0x7FC00000,0.000000e+00), (0x7FC00000,1.000000e+00)], [(0x7FC00000,2.000000e+00), (0x7FC00000,3.000000e+00)],
    [(0x7FC00000,4.000000e+00), (0x7FC00000,5.000000e+00)], [(0x7FC00000,6.000000e+00), (0x7FC00000,7.000000e+00)]
  ]> : tensor<4x2xcomplex<f32>>
  %empty = tensor.empty() : tensor<4x2xcomplex<f32>>
  %0 = flow.dispatch.region -> (tensor<4x2xcomplex<f32>>) {
    %generic = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %const : tensor<4xi32>, tensor<4x2xcomplex<f32>>) outs(%empty : tensor<4x2xcomplex<f32>>) {
    ^bb0(%in0: i32, %in1: complex<f32>, %out: complex<f32>):
      %i1 = linalg.index 1 : index
      %i0 = arith.index_cast %in0 : i32 to index
      %extract = tensor.extract %table[%i0, %i1] : tensor<8x2xcomplex<f32>>
      %cmp = arith.cmpi sle, %in0, %c4095 : i32
      %select = arith.select %cmp, %extract, %in1 : complex<f32>
      linalg.yield %select : complex<f32>
    } -> tensor<4x2xcomplex<f32>>
    flow.return %generic : tensor<4x2xcomplex<f32>>
  }
  return %0 : tensor<4x2xcomplex<f32>>
}

// CHECK-LABEL: func.func @complex_element_type
//       CHECK:   flow.dispatch.region
//       CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<4x2xcomplex<f32>>
//       CHECK:     %[[CST:.+]] = arith.constant dense<{{.+}}> : tensor<4x2xcomplex<f32>>
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%{{.+}}, %[[CST]] : tensor<4xi32>, tensor<4x2xcomplex<f32>>)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<4x2xcomplex<f32>>)
//       CHECK:   flow.return
