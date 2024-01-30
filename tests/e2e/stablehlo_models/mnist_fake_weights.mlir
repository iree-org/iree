// MNIST model with placeholder weights, for testing.

// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=vmvx %s --input=1x28x28x1xf32 | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=llvm-cpu %s --input=1x28x28x1xf32 | FileCheck %s

module {
  util.global private @"__iree_flow___sm_node17__model.layer-1.kernel" {inlining_policy = #util.inline.never} = #util.byte_pattern<1> : tensor<784x128xf32>
  util.global private @"__iree_flow___sm_node18__model.layer-1.bias" {inlining_policy = #util.inline.never} = #util.byte_pattern<2> : tensor<128xf32>
  util.global private @"__iree_flow___sm_node24__model.layer-2.kernel" {inlining_policy = #util.inline.never} = #util.byte_pattern<3> : tensor<128x10xf32>
  util.global private @"__iree_flow___sm_node25__model.layer-2.bias" {inlining_policy = #util.inline.never} = #util.byte_pattern<4> : tensor<10xf32>
  func.func @predict(%arg0: tensor<1x28x28x1xf32>) -> tensor<1x10xf32> attributes {iree.module.export, iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I8!S5!k0_0R3!_0"}} {
    %ptr___iree_flow___sm_node17__model.layer-1.kernel = util.global.address @"__iree_flow___sm_node17__model.layer-1.kernel" : !util.ptr<tensor<784x128xf32>>
    %ptr___iree_flow___sm_node18__model.layer-1.bias = util.global.address @"__iree_flow___sm_node18__model.layer-1.bias" : !util.ptr<tensor<128xf32>>
    %ptr___iree_flow___sm_node24__model.layer-2.kernel = util.global.address @"__iree_flow___sm_node24__model.layer-2.kernel" : !util.ptr<tensor<128x10xf32>>
    %ptr___iree_flow___sm_node25__model.layer-2.bias = util.global.address @"__iree_flow___sm_node25__model.layer-2.bias" : !util.ptr<tensor<10xf32>>
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x128xf32>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = util.global.load.indirect %ptr___iree_flow___sm_node25__model.layer-2.bias : !util.ptr<tensor<10xf32>> -> tensor<10xf32>
    %4 = util.global.load.indirect %ptr___iree_flow___sm_node24__model.layer-2.kernel : !util.ptr<tensor<128x10xf32>> -> tensor<128x10xf32>
    %5 = util.global.load.indirect %ptr___iree_flow___sm_node18__model.layer-1.bias : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
    %6 = util.global.load.indirect %ptr___iree_flow___sm_node17__model.layer-1.kernel : !util.ptr<tensor<784x128xf32>> -> tensor<784x128xf32>
    %7 = stablehlo.reshape %arg0 : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
    %8 = stablehlo.dot %7, %6 : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
    %9 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128xf32>
    %11 = stablehlo.maximum %10, %0 : tensor<1x128xf32>
    %12 = stablehlo.dot %11, %4 : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
    %13 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x10xf32>
    %15 = stablehlo.reduce(%14 init: %1) across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %22 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %22 : tensor<f32>
    }
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<1xf32>) -> tensor<1x10xf32>
    %17 = stablehlo.subtract %14, %16 : tensor<1x10xf32>
    %18 = stablehlo.exponential %17 : tensor<1x10xf32>
    %19 = stablehlo.reduce(%18 init: %2) across dimensions = [1] : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %22 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %22 : tensor<f32>
    }
    %20 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1xf32>) -> tensor<1x10xf32>
    %21 = stablehlo.divide %18, %20 : tensor<1x10xf32>
    return %21 : tensor<1x10xf32>
  }
}

// CHECK: 1x10xf32=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
