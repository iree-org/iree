// MNIST model with placeholder weights, for translation testing.

// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s -input-value="1x28x28x1xf32" | IreeFileCheck %s
// RUN: iree-run-mlir -iree-hal-target-backends=llvm-ir %s -input-value="1x28x28x1xf32" | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value="1x28x28x1xf32" | IreeFileCheck %s)

module {
  // CHECK-LABEL: EXEC @main
  func @main(%arg0: tensor<1x28x28x1xf32>) -> tuple<tensor<1x10xf32>>
  attributes {iree.module.export} {
    %cst = constant  {name = "constant.9"} dense<0.5> : tensor<f32>
    %0 = "xla_hlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[]> : tensor<0xi64>, name = "broadcast.10"} : (tensor<f32>) -> tensor<1x128xf32>
    %1 = "xla_hlo.copy"(%arg0) {name = "copy.1"} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28x1xf32>
    %2 = "xla_hlo.reshape"(%1) {name = "reshape.2"} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28x1xf32>
    %3 = "xla_hlo.reshape"(%2) {name = "reshape.3"} : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
    %cst_0 = constant  {name = "constant.4"} dense<0.5> : tensor<784x128xf32>
    %4 = "xla_hlo.dot"(%3, %cst_0) {name = "dot.5", precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
    %cst_1 = constant  {name = "constant.6"} dense<0.5> : tensor<128xf32>
    %5 = "xla_hlo.broadcast_in_dim"(%cst_1) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.7"} : (tensor<128xf32>) -> tensor<1x128xf32>
    %6 = "xla_hlo.add"(%4, %5) {name = "add.8"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %7 = "xla_hlo.maximum"(%0, %6) {name = "maximum.11"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %cst_2 = constant  {name = "constant.12"} dense<0.5> : tensor<128x10xf32>
    %8 = "xla_hlo.dot"(%7, %cst_2) {name = "dot.13", precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
    %cst_3 = constant  {name = "constant.14"} dense<0.5> : tensor<10xf32>
    %9 = "xla_hlo.broadcast_in_dim"(%cst_3) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.15"} : (tensor<10xf32>) -> tensor<1x10xf32>
    %10 = "xla_hlo.add"(%8, %9) {name = "add.16"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %cst_4 = constant  {name = "constant.17"} dense<0xFF800000> : tensor<f32>
    %11 = "xla_hlo.reduce"(%10, %cst_4) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):   // no predecessors
      %20 = "xla_hlo.maximum"(%arg1, %arg2) {name = "maximum.21"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%20) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %12 = "xla_hlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<0> : tensor<1xi64>, name = "broadcast.23"} : (tensor<1xf32>) -> tensor<1x10xf32>
    %13 = "xla_hlo.subtract"(%10, %12) {name = "subtract.24"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = "xla_hlo.exp"(%13) {name = "exponential.25"} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %cst_5 = constant  {name = "constant.27"} dense<0.5> : tensor<f32>
    %15 = "xla_hlo.reduce"(%14, %cst_5) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):   // no predecessors
      %21 = "xla_hlo.add"(%arg3, %arg4) {name = "add.31"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%21) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %16 = "xla_hlo.broadcast_in_dim"(%15) {broadcast_dimensions = dense<0> : tensor<1xi64>, name = "broadcast.34"} : (tensor<1xf32>) -> tensor<1x10xf32>
    %17 = "xla_hlo.divide"(%14, %16) {name = "divide.35"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %18 = "xla_hlo.reshape"(%17) {name = "reshape.36"} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %19 = "xla_hlo.tuple"(%18) {name = "tuple.37"} : (tensor<1x10xf32>) -> tuple<tensor<1x10xf32>>
    return %19 : tuple<tensor<1x10xf32>>
  }
}

// CHECK: 1x10xf32=[0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}} 0.09{{[0-9]+}}]
