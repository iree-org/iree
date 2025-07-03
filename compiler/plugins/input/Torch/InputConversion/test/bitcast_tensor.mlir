// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(torch-iree-bitcast-tensor))" -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward
func.func @forward(%arg0: !torch.vtensor<[1,1,8],f16>) -> !torch.vtensor<[1,1,8],f16> {
  %q_rhs = torch.vtensor.literal(dense<[[57, 128, 249, 244], [7, 243, 27, 15], [1, 2, 159, 71], [159, 253, 160, 231], [248, 224, 191, 228], [96, 15, 158, 220], [240, 250, 47, 208], [127, 192, 239, 176]]> : tensor<8x4xui8>) : !torch.vtensor<[8,4],ui8>
  // CHECK: %[[C0:.*]] = torch.vtensor.literal(dense<{{\[\[}}57, 128, 249, 244], [7, 243, 27, 15], [1, 2, 159, 71], [159, 253, 160, 231], [248, 224, 191, 228], [96, 15, 158, 220], [240, 250, 47, 208], [127, 192, 239, 176]]> : tensor<8x4xui8>) : !torch.vtensor<[8,4],ui8>
  %scales = torch.vtensor.literal(dense<1.0> : tensor<8x4x1xf16>) : !torch.vtensor<[8,4,1],f16>
  %zps = torch.vtensor.literal(dense<0.0> : tensor<8x4x1xf16>) : !torch.vtensor<[8,4,1],f16>
  %bit_width = torch.constant.int 4
  %group_size = torch.constant.int 2
  // CHECK: %[[TOBUILTIN:.*]] = torch_c.to_builtin_tensor %[[C0]] : !torch.vtensor<[8,4],ui8> -> tensor<8x4xui8>
  // CHECK: %[[BITCAST:.*]] = iree_tensor_ext.bitcast %[[TOBUILTIN]] : tensor<8x4xui8> -> tensor<8x8xi4>
  // CHECK: %[[TOTORCH:.*]] = torch_c.from_builtin_tensor %[[BITCAST]] : tensor<8x8xi4> -> !torch.vtensor<[8,8],ui4>
  %output = torch.operator "quant.matmul_rhs_group_quant"(%arg0, %q_rhs, %scales, %zps, %bit_width, %group_size) : (!torch.vtensor<[1,1,8],f16>, !torch.vtensor<[8,4],ui8>, !torch.vtensor<[8,4,1],f16>, !torch.vtensor<[8,4,1],f16>, !torch.int, !torch.int) -> !torch.vtensor<[1,1,8],f16>
  return %output : !torch.vtensor<[1,1,8],f16>
}

// -----

// CHECK-LABEL: @view_type0
func.func @view_type0(%arg0 : !torch.vtensor<[295501824],ui8>) -> !torch.vtensor<[147750912],si16> {
  %int4 = torch.constant.int 4
  // CHECK: iree_tensor_ext.bitcast %[[IN:.+]] : tensor<295501824xi8> -> tensor<147750912xi16>
  %0 = torch.aten.view.dtype %arg0, %int4 : !torch.vtensor<[295501824],ui8>, !torch.int -> !torch.vtensor<[147750912],si16>
  return %0 : !torch.vtensor<[147750912],si16>
}

// -----

// CHECK-LABEL: @view_type1
func.func @view_type1(%arg0 : !torch.vtensor<[?],ui8>) -> !torch.vtensor<[?],si16> {
  %int4 = torch.constant.int 4
  //  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  //      CHECK: %[[DIM:.+]] = tensor.dim %[[IN:.+]], %[[C0]] : tensor<?xi8>
  //      CHECK: %[[OUT:.+]] = arith.muli %[[DIM]], %[[C2]]
  //      CHECK: iree_tensor_ext.bitcast %[[IN]] :
  // CHECK-SAME:   tensor<?xi8>{%[[DIM]]} -> tensor<?xi16>{%[[OUT]]}
  %0 = torch.aten.view.dtype %arg0, %int4 : !torch.vtensor<[?],ui8>, !torch.int -> !torch.vtensor<[?],si16>
  return %0 : !torch.vtensor<[?],si16>
}

// -----

// CHECK-LABEL: @view_type2
func.func @view_type2(%arg0 : !torch.vtensor<[?],f64>) -> !torch.vtensor<[?],f16> {
  %int4 = torch.constant.int 4
  //  CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  //      CHECK: %[[DIM:.+]] = tensor.dim %[[IN:.+]], %[[C0]] : tensor<?xf64>
  //      CHECK: %[[OUT:.+]] = arith.divsi %[[DIM]], %[[C4]] : index
  //      CHECK: iree_tensor_ext.bitcast %[[IN]] :
  // CHECK-SAME:   tensor<?xf64>{%[[DIM]]} -> tensor<?xf16>{%[[OUT]]}
  %0 = torch.aten.view.dtype %arg0, %int4 : !torch.vtensor<[?],f64>, !torch.int -> !torch.vtensor<[?],f16>
  return %0 : !torch.vtensor<[?],f16>
}

// -----

// CHECK-LABEL: @view_type3
func.func @view_type3(%arg0 : !torch.vtensor<[?,295501824],ui8>) -> !torch.vtensor<[?,147750912],si16> {
  %int4 = torch.constant.int 4
  //      CHECK: %[[C0:.+]] = arith.constant 0 : index
  //      CHECK: %[[DIM:.+]] = tensor.dim %[[IN:.+]], %[[C0]] : tensor<?x295501824xi8>
  //      CHECK: iree_tensor_ext.bitcast %[[IN]] :
  // CHECK-SAME:   tensor<?x295501824xi8>{%[[DIM]]} -> tensor<?x147750912xi16>{%[[DIM]]}
  %0 = torch.aten.view.dtype %arg0, %int4 : !torch.vtensor<[?,295501824],ui8>, !torch.int -> !torch.vtensor<[?,147750912],si16>
  return %0 : !torch.vtensor<[?,147750912],si16>
}

// -----

// CHECK-LABEL: @view_as_complex
func.func @view_as_complex(%arg0 : !torch.vtensor<[128,2],f32>) -> !torch.vtensor<[128], complex<f32>> {
  // CHECK: iree_tensor_ext.bitcast %[[IN:.+]] : tensor<128x2xf32> -> tensor<128xcomplex<f32>>
  %0 = torch.aten.view_as_complex %arg0 : !torch.vtensor<[128,2],f32> -> !torch.vtensor<[128],complex<f32>>
  return %0 : !torch.vtensor<[128],complex<f32>>
}

// -----

// CHECK-LABEL: @view_as_real
func.func @view_as_real(%arg0 : !torch.vtensor<[128], complex<f32>>) -> !torch.vtensor<[128,2],f32> {
  // CHECK: iree_tensor_ext.bitcast %[[IN:.+]] : tensor<128xcomplex<f32>> -> tensor<128x2xf32>
  %0 = torch.aten.view_as_real %arg0 : !torch.vtensor<[128],complex<f32>> -> !torch.vtensor<[128,2],f32>
  return %0 : !torch.vtensor<[128,2],f32>
}
