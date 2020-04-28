// MNIST model with placeholder weights, for testing.

// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s -input-value="1x28x28x1xf32" | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=llvm-ir %s -input-value="1x28x28x1xf32" | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value="1x28x28x1xf32" | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -iree-use-linalg-to-spirv-path %s -input-value="1x28x28x1xf32" | IreeFileCheck %s)

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 175 : i32}} {
  flow.variable @"__iree_flow___sm_node15__model.layer-2.kernel" dense<0.5> : tensor<784x128xf32>
  flow.variable @"__iree_flow___sm_node16__model.layer-2.bias" dense<0.1> : tensor<128xf32>
  flow.variable @"__iree_flow___sm_node21__model.layer-3.kernel" dense<0.5> : tensor<128x10xf32>
  flow.variable @"__iree_flow___sm_node22__model.layer-3.bias" dense<0.1> : tensor<10xf32>
  // CHECK-LABEL: EXEC @predict
  func @predict(%arg0: tensor<1x28x28x1xf32>) -> tensor<1x10xf32> attributes {iree.module.export, iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I8!S5!k0_0R3!_0"}, tf._input_shapes = ["tfshape$dim { size: 1 } dim { size: 28 } dim { size: 28 } dim { size: 1 }", "tfshape$unknown_rank: true", "tfshape$unknown_rank: true", "tfshape$unknown_rank: true", "tfshape$unknown_rank: true"], tf.signature.is_stateful} {
    %0 = flow.variable.address @"__iree_flow___sm_node15__model.layer-2.kernel" : !iree.ptr<tensor<784x128xf32>>
    %1 = flow.variable.address @"__iree_flow___sm_node16__model.layer-2.bias" : !iree.ptr<tensor<128xf32>>
    %2 = flow.variable.address @"__iree_flow___sm_node21__model.layer-3.kernel" : !iree.ptr<tensor<128x10xf32>>
    %3 = flow.variable.address @"__iree_flow___sm_node22__model.layer-3.bias" : !iree.ptr<tensor<10xf32>>
    %4 = xla_hlo.constant dense<0xFF800000> : tensor<f32>
    %5 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = flow.variable.load.indirect %3 : !iree.ptr<tensor<10xf32>> -> tensor<10xf32>
    %7 = flow.variable.load.indirect %2 : !iree.ptr<tensor<128x10xf32>> -> tensor<128x10xf32>
    %8 = flow.variable.load.indirect %1 : !iree.ptr<tensor<128xf32>> -> tensor<128xf32>
    %9 = flow.variable.load.indirect %0 : !iree.ptr<tensor<784x128xf32>> -> tensor<784x128xf32>
    %10 = "xla_hlo.reshape"(%arg0) : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
    %11 = "xla_hlo.dot"(%10, %9) : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
    %12 = "xla_hlo.add"(%11, %8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>
    %13 = "xla_hlo.maximum"(%12, %5) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x128xf32>
    %14 = "xla_hlo.dot"(%13, %7) : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
    %15 = "xla_hlo.add"(%14, %6) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    %16 = "xla_hlo.reduce"(%15, %4) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):   // no predecessors
      %21 = xla_hlo.maximum %arg1, %arg2 : tensor<f32>
      "xla_hlo.return"(%21) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %17 = "xla_hlo.subtract"(%15, %16) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1xf32>) -> tensor<1x10xf32>
    %18 = "xla_hlo.exponential"(%17) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %19 = "xla_hlo.reduce"(%18, %5) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):   // no predecessors
      %21 = xla_hlo.add %arg1, %arg2 : tensor<f32>
      "xla_hlo.return"(%21) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %20 = "xla_hlo.divide"(%18, %19) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1xf32>) -> tensor<1x10xf32>
    return %20 : tensor<1x10xf32>
  }
}

// CHECK: 1x10xf32=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
