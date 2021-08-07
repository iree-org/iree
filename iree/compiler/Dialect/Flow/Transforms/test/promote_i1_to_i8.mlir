
// RUN: iree-opt -split-input-file -pass-pipeline='builtin.func(iree-flow-promote-i1-to-i8)' %s | IreeFileCheck %s

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: boolean_const
func @boolean_const() -> (tensor<4xi1>) {
    // CHECK: [[CONST:%.+]] = constant dense<[1, 1, 0, 1]> : tensor<4xi8>
    // CHECK: [[INIT:%.+]] = linalg.init_tensor [4] : tensor<4xi1>
    // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]} ins([[CONST]] : tensor<4xi8>) outs([[INIT]] : tensor<4xi1>)
    // CHECK: ^bb0(%arg0: i8, %arg1: i1):
    // CHECK:   [[TRUNC:%.+]] = trunci %arg0 : i8 to i1
    // CHECK:   linalg.yield [[TRUNC]]
    // CHECK: return [[GENERIC]]
    %0 = constant dense<[true, true, false, true]> : tensor<4xi1>
    return %0 : tensor<4xi1>
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: boolean_const
func @boolean_const() -> (tensor<4xi1>, tensor<4xi8>) {
    // CHECK: [[CONST:%.+]] = constant dense<[1, 1, 0, 1]> : tensor<4xi8>
    // CHECK: [[INIT:%.+]] = linalg.init_tensor [4] : tensor<4xi1>
    // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]], iterator_types = ["parallel"]} ins([[CONST]] : tensor<4xi8>) outs([[INIT]] : tensor<4xi1>)
    // CHECK: ^bb0(%arg0: i8, %arg1: i1):
    // CHECK:   [[TRUNC:%.+]] = trunci %arg0 : i8 to i1
    // CHECK:   linalg.yield [[TRUNC]]
    // CHECK: return [[GENERIC]], [[CONST]]
    %0 = constant dense<[true, true, false, true]> : tensor<4xi1>
    %1 = constant dense<[1, 1, 0, 1]> : tensor<4xi8>
    return %0, %1 : tensor<4xi1>, tensor<4xi8>
}
