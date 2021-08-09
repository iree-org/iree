// MNIST model with placeholder weights, for testing.

// RUN: iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vmvx %s -function-input="1x28x28x1xf32" | IreeFileCheck %s
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=dylib-llvm-aot %s -function-input="1x28x28x1xf32" | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vulkan-spirv %s -function-input="1x28x28x1xf32" | IreeFileCheck %s)

module {
  flow.variable @"__iree_flow___sm_node17__model.layer-1.kernel" dense<1.000000e+00> : tensor<784x128xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node18__model.layer-1.bias" dense<5.000000e-01> : tensor<128xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node24__model.layer-2.kernel" dense<0.333333343> : tensor<128x10xf32> attributes {noinline, sym_visibility = "private"}
  flow.variable @"__iree_flow___sm_node25__model.layer-2.bias" dense<2.500000e-01> : tensor<10xf32> attributes {noinline, sym_visibility = "private"}
  // CHECK-LABEL: EXEC @predict
  func @predict(%arg0: tensor<1x28x28x1xf32>) -> tensor<1x10xf32> attributes {iree.module.export, iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I8!S5!k0_0R3!_0"}} {
    %0 = flow.variable.address @"__iree_flow___sm_node17__model.layer-1.kernel" : !util.ptr<tensor<784x128xf32>>
    %1 = flow.variable.address @"__iree_flow___sm_node18__model.layer-1.bias" : !util.ptr<tensor<128xf32>>
    %2 = flow.variable.address @"__iree_flow___sm_node24__model.layer-2.kernel" : !util.ptr<tensor<128x10xf32>>
    %3 = flow.variable.address @"__iree_flow___sm_node25__model.layer-2.bias" : !util.ptr<tensor<10xf32>>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<1x128xf32>
    %5 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %6 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = flow.variable.load.indirect %3 : !util.ptr<tensor<10xf32>> -> tensor<10xf32>
    %8 = flow.variable.load.indirect %2 : !util.ptr<tensor<128x10xf32>> -> tensor<128x10xf32>
    %9 = flow.variable.load.indirect %1 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %10 = flow.variable.load.indirect %0 : !util.ptr<tensor<784x128xf32>> -> tensor<784x128xf32>
    %11 = "mhlo.reshape"(%arg0) : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
    %12 = "mhlo.dot"(%11, %10) : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
    %13 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128xf32>
    %14 = mhlo.add %12, %13 : tensor<1x128xf32>
    %15 = mhlo.maximum %14, %4 : tensor<1x128xf32>
    %16 = "mhlo.dot"(%15, %8) : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
    %17 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>) -> tensor<1x10xf32>
    %18 = mhlo.add %16, %17 : tensor<1x10xf32>
    %19 = "mhlo.reduce"(%18, %5) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %26 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%26) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %20 = "mhlo.broadcast_in_dim"(%19) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
    %21 = mhlo.subtract %18, %20 : tensor<1x10xf32>
    %22 = "mhlo.exponential"(%21) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %23 = "mhlo.reduce"(%22, %6) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %26 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%26) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %24 = "mhlo.broadcast_in_dim"(%23) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
    %25 = mhlo.divide %22, %24 : tensor<1x10xf32>
    return %25 : tensor<1x10xf32>
  }
}

// CHECK: 1x10xf32=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
