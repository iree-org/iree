// RUN: iree-opt %s -iree-make-executable-abi -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @staticOutputEntry
func @staticOutputEntry(%arg0: memref<4x2xf32>, %arg1: memref<4x2xf32>)
    // CHECK: attributes
    attributes {iree.executable.export} {
  // CHECK-NEXT: %0 = iree.memref_to_tensor(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  %0 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  // CHECK-NEXT: %1 = call @staticOutput(%0) : (tensor<4x2xf32>) -> tensor<4x2xf32>
  %1 = call @staticOutput(%0) : (tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: %2 = iree.tensor_to_memref(%1 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK-NEXT: %3 = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: %4 = iree.constant[dense<[4, 2]> : tensor<2xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%2, %3, %arg1, %3, %4)
  iree.store_output(%1 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  // CHECK-NEXT: return
  return
}

func @staticOutput(%arg0 : tensor<4x2xf32>) -> tensor<4x2xf32>
    attributes {iree.dispatchable} {
  return %arg0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: func @scalarFnEntry
func @scalarFnEntry(%arg0: memref<f32>, %arg1: memref<f32>)
    // CHECK: attributes
    attributes {iree.executable.export} {
  // CHECK-NEXT: %0 = load %arg0[] : memref<f32>
  %0 = iree.load_input(%arg0 : memref<f32>) : f32
  // CHECK-NEXT: %1 = call @scalarFn(%0) : (f32) -> f32
  %1 = call @scalarFn(%0) : (f32) -> f32
  // CHECK-NEXT: store %1, %arg1[] : memref<f32>
  iree.store_output(%1 : f32, %arg1 : memref<f32>)
  // CHECK-NEXT: return
  return
}

func @scalarFn(%arg0 : f32) -> f32
    attributes {iree.dispatchable} {
  return %arg0 : f32
}

// -----

// CHECK-LABEL: func @scalarTensorFnEntry
func @scalarTensorFnEntry(%arg0: memref<f32>, %arg1: memref<f32>)
    // CHECK: attributes
    attributes {iree.executable.export} {
  // CHECK-NEXT: %0 = iree.memref_to_tensor(%arg0 : memref<f32>) : tensor<f32>
  %0 = iree.load_input(%arg0 : memref<f32>) : tensor<f32>
  // CHECK-NEXT: %1 = call @scalarTensorFn(%0) : (tensor<f32>) -> tensor<f32>
  %1 = call @scalarTensorFn(%0) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: %2 = iree.tensor_to_memref(%1 : tensor<f32>) : memref<f32>
  // CHECK-NEXT: %3 = iree.constant[dense<[]> : tensor<0xi64>
  // CHECK-NEXT: %4 = iree.constant[dense<[]> : tensor<0xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%2, %3, %arg1, %3, %4) : (memref<f32>, memref<0xi64>, memref<f32>, memref<0xi64>, memref<0xi64>) -> ()
  iree.store_output(%1 : tensor<f32>, %arg1 : memref<f32>)
  // CHECK-NEXT: return
  return
}

func @scalarTensorFn(%arg0 : tensor<f32>) -> tensor<f32>
    attributes {iree.dispatchable} {
  return %arg0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @returnValuesEntry
func @returnValuesEntry(%arg0: memref<4x2xf32>, %arg1: memref<4x2xf32>, %arg2: memref<4x2xf32>) -> (memref<4x2xf32>, memref<4x2xf32>)
    // CHECK: attributes
    attributes {iree.executable.export} {
  // CHECK-NEXT: %0 = iree.memref_to_tensor(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  %0 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  // CHECK-NEXT: call @returnValues
  %1, %2 = call @returnValues(%0) : (tensor<4x2xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>)
  // CHECK-NEXT: %2 = iree.tensor_to_memref(%1#0 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK-NEXT: %3 = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: %4 = iree.constant[dense<[4, 2]> : tensor<2xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%2, %3, %arg1, %3, %4)
  iree.store_output(%1 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  // CHECK-NEXT: %5 = iree.tensor_to_memref(%1#1 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK-NEXT: %6 = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: %7 = iree.constant[dense<[4, 2]> : tensor<2xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%5, %6, %arg2, %6, %7)
  iree.store_output(%2 : tensor<4x2xf32>, %arg2 : memref<4x2xf32>)
  %3 = iree.tensor_to_memref(%1 : tensor<4x2xf32>) : memref<4x2xf32>
  %4 = iree.tensor_to_memref(%2 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK: return %8, %9 : memref<4x2xf32>, memref<4x2xf32>
  return %3, %4 : memref<4x2xf32>, memref<4x2xf32>
}

func @returnValues(%arg0 : tensor<4x2xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>)
    attributes {iree.dispatchable} {
  return %arg0, %arg0 : tensor<4x2xf32>, tensor<4x2xf32>
}

// -----

// CHECK-LABEL: func @aliasInputsEntry
func @aliasInputsEntry(%arg0: memref<4x2xf32>, %arg1: memref<4x2xf32>)
    // CHECK: attributes
    attributes {iree.executable.export} {
  // CHECK-NEXT: %0 = iree.memref_to_tensor(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  %0 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  // CHECK-NEXT: %1 = iree.memref_to_tensor(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  %1 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  // CHECK-NEXT: call @aliasInputs
  %2 = call @aliasInputs(%0, %1) : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: %3 = iree.tensor_to_memref(%2 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK-NEXT: %4 = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: %5 = iree.constant[dense<[4, 2]> : tensor<2xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%3, %4, %arg1, %4, %5)
  iree.store_output(%2 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  // CHECK-NEXT: return
  return
}

func @aliasInputs(%arg0 : tensor<4x2xf32>, %arg1 : tensor<4x2xf32>) -> tensor<4x2xf32>
    attributes {iree.dispatchable} {
  return %arg0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: func @aliasOutputsEntry
func @aliasOutputsEntry(%arg0: memref<4x2xf32>, %arg1: memref<4x2xf32>)
    // CHECK: attributes
    attributes {iree.executable.export} {
  // CHECK-NEXT: %0 = iree.memref_to_tensor(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  %0 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  // CHECK-NEXT: call @aliasOutputs
  %1 = call @aliasOutputs(%0) : (tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: %2 = iree.tensor_to_memref(%1 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK-NEXT: %3 = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: %4 = iree.constant[dense<[4, 2]> : tensor<2xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%2, %3, %arg1, %3, %4)
  iree.store_output(%1 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  // CHECK-NEXT: %5 = iree.tensor_to_memref(%1 : tensor<4x2xf32>) : memref<4x2xf32>
  // CHECK-NEXT: %6 = iree.constant[dense<0> : tensor<2xi64>
  // CHECK-NEXT: %7 = iree.constant[dense<[4, 2]> : tensor<2xi64>
  // CHECK-NEXT: "iree_hl_interp.copy"(%5, %6, %arg1, %6, %7)
  iree.store_output(%1 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  // CHECK-NEXT: return
  return
}

func @aliasOutputs(%arg0 : tensor<4x2xf32>) -> tensor<4x2xf32>
    attributes {iree.dispatchable} {
  return %arg0 : tensor<4x2xf32>
}
