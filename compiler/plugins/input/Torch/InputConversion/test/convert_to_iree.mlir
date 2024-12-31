// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(torch-convert-to-iree))" -split-input-file -verify-diagnostics | FileCheck %s

func.func @view_as_real(%arg0: !torch.vtensor<[8],complex<f32>>) -> !torch.vtensor<[8,2],f32> {
    %0 = torch.aten.view_as_real %arg0 : !torch.vtensor<[8],complex<f32>> -> !torch.vtensor<[8,2],f32>
    return %0 : !torch.vtensor<[8,2],f32>
}

// CHECK-LABEL: @view_as_real
//  CHECK-SAME:   %[[ARG0:.+]]: !torch.vtensor<[8],complex<f32>>
//       CHECK:   %[[IN:.+]] = torch_c.to_builtin_tensor %[[ARG0]]
//       CHECK:   %[[BITCAST:.+]] = flow.tensor.bitcast %[[IN]] : tensor<8xcomplex<f32>> -> tensor<8x2xf32>
//       CHECK:   %[[RES:.+]] = torch_c.from_builtin_tensor %[[BITCAST]]
//       CHECK:   return %[[RES]]

// -----

func.func @view_as_complex(%arg0: !torch.vtensor<[8,2],f32>) -> !torch.vtensor<[8],complex<f32>> {
    %0 = torch.aten.view_as_complex %arg0 : !torch.vtensor<[8,2],f32> -> !torch.vtensor<[8],complex<f32>>
    return %0 : !torch.vtensor<[8],complex<f32>>
}

// CHECK-LABEL: @view_as_complex
//  CHECK-SAME:   %[[ARG0:.+]]: !torch.vtensor<[8,2],f32>
//       CHECK:   %[[IN:.+]] = torch_c.to_builtin_tensor %[[ARG0]]
//       CHECK:   %[[BITCAST:.+]] = flow.tensor.bitcast %[[IN]] : tensor<8x2xf32> -> tensor<8xcomplex<f32>>
//       CHECK:   %[[RES:.+]] = torch_c.from_builtin_tensor %[[BITCAST]]
//       CHECK:   return %[[RES]]

// -----

func.func @view_as_real_dynamic(%arg0: !torch.vtensor<[?],complex<f32>>) -> !torch.vtensor<[?,2],f32> {
    %0 = torch.aten.view_as_real %arg0 : !torch.vtensor<[?],complex<f32>> -> !torch.vtensor<[?,2],f32>
    return %0 : !torch.vtensor<[?,2],f32>
}

// CHECK-LABEL: @view_as_real_dynamic
//  CHECK-SAME:   %[[ARG0:.+]]: !torch.vtensor<[?],complex<f32>>
//       CHECK:   %[[IN:.+]] = torch_c.to_builtin_tensor %[[ARG0]]
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[IN]], %c0 : tensor<?xcomplex<f32>>
//       CHECK:   %[[BITCAST:.+]] = flow.tensor.bitcast %[[IN]] : tensor<?xcomplex<f32>>{%[[DIM]]} -> tensor<?x2xf32>{%[[DIM]]}
//       CHECK:   %[[RES:.+]] = torch_c.from_builtin_tensor %[[BITCAST]]
//       CHECK:   return %[[RES]]
