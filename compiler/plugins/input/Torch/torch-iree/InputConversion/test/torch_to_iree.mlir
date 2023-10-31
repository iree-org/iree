// RUN: iree-opt --split-input-file --torch-to-iree %s | FileCheck %s
// This is just a smoke test that the pipeline functions, and it also
// validates any features different from upstream.

// Verify that we can have IREE ops in the input and the types convert
// properly.
// CHECK: func @forward(%arg0: tensor<128x20xf32>) -> tensor<128x30xf32>
// CHECK: linalg.matmul
module {
  func.func @forward(%arg0: !torch.vtensor<[128,20],f32>) -> !torch.vtensor<[128,30],f32> {
    %_params.classifier.weight = util.global.load @_params.classifier.weight : tensor<30x20xf32>
    %0 = torch_c.from_builtin_tensor %_params.classifier.weight : tensor<30x20xf32> -> !torch.vtensor<[30,20],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %1 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[30,20],f32>, !torch.int, !torch.int -> !torch.vtensor<[20,30],f32>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[128,20],f32>, !torch.vtensor<[20,30],f32> -> !torch.vtensor<[128,30],f32>
    %int1_0 = torch.constant.int 1
    %3 = torch.aten.mul.Scalar %2, %int1_0 : !torch.vtensor<[128,30],f32>, !torch.int -> !torch.vtensor<[128,30],f32>
    %_params.classifier.bias = util.global.load @_params.classifier.bias : tensor<30xf32>
    %4 = torch_c.from_builtin_tensor %_params.classifier.bias : tensor<30xf32> -> !torch.vtensor<[30],f32>
    %int1_1 = torch.constant.int 1
    %5 = torch.aten.mul.Scalar %4, %int1_1 : !torch.vtensor<[30],f32>, !torch.int -> !torch.vtensor<[30],f32>
    %int1_2 = torch.constant.int 1
    %6 = torch.aten.add.Tensor %3, %5, %int1_2 : !torch.vtensor<[128,30],f32>, !torch.vtensor<[30],f32>, !torch.int -> !torch.vtensor<[128,30],f32>
    return %6 : !torch.vtensor<[128,30],f32>
  }
  util.global private @_params.classifier.weight {noinline} : tensor<30x20xf32>
  util.global private @_params.classifier.bias {noinline} : tensor<30xf32>
}

// -----

// Verify we can decompose complex ops
// CHECK: func @main(%arg0: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>) 
// CHECK: tensor.empty
module {
  func.func @main(%arg0: !torch.vtensor<[2,3,4],f32>) -> (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>) {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
    %0 = torch.prim.ListConstruct %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int12 = torch.constant.int 12
    %int4_0 = torch.constant.int 4
    %int1 = torch.constant.int 1
    %1 = torch.prim.ListConstruct %int12, %int4_0, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %none = torch.constant.none
    %none_1 = torch.constant.none    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %2 = torch.aten.empty_strided %0, %1, %none, %none_1, %cpu, %false : !torch.list<int>, !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2,3,4],f32>
    %false_2 = torch.constant.bool false
    %3 = torch.aten.copy %arg0, %2, %false_2 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.bool -> !torch.vtensor<[2,3,4],f32>
    return %3, %3 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>
  }
}
