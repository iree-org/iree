// RUN: iree-opt --split-input-file --torch-to-iree %s | FileCheck %s --check-prefixes=CHECK,NONFINITE
// RUN: iree-opt --split-input-file --torch-to-iree="allow-non-finites=false" %s | FileCheck %s --check-prefixes=CHECK,FINITE
// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s --check-prefixes=CHECK,NONFINITE
// RUN: iree-compile --compile-to=input --split-input-file --iree-torch-allow-non-finites=false %s | FileCheck %s --check-prefixes=CHECK,FINITE
// This is just a smoke test that the pipeline functions, and it also
// validates any features different from upstream.

// Verify that we can have IREE ops in the input and the types convert
// properly.
// CHECK-LABEL: util.func public @forward$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view
// CHECK: linalg.matmul
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
util.global private @_params.classifier.weight <inlining_policy = #util.inline.never> : tensor<30x20xf32>
util.global private @_params.classifier.bias <inlining_policy = #util.inline.never> : tensor<30xf32>

// -----

// Verify we can decompose complex ops
// CHECK-LABEL: util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)
// CHECK: tensor.empty
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
  %none_1 = torch.constant.none
  %cpu = torch.constant.device "cpu"
  %false = torch.constant.bool false
  %2 = torch.aten.empty_strided %0, %1, %none, %none_1, %cpu, %false : !torch.list<int>, !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2,3,4],f32>
  %false_2 = torch.constant.bool false
  %3 = torch.aten.copy %arg0, %2, %false_2 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>, !torch.bool -> !torch.vtensor<[2,3,4],f32>
  return %3, %3 : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[2,3,4],f32>
}

// -----

// The pipeline converts PT2E quantization to LinalgExt before Torch-to-Linalg
// would expand it into generic arithmetic.
// CHECK-LABEL: util.func public @dequantize$async(
// CHECK: iree_linalg_ext.dequantize_affine
func.func @dequantize(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %out_dtype = torch.derefine %none : !torch.none to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.none,
        !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// CHECK-LABEL: util.func public @amax$async(
// NONFINITE: arith.constant 0xFF800000 : f32
// FINITE: arith.constant -3.40282347E+38 : f32
func.func @amax(%input: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4],f32> {
  %dim = torch.constant.int 1
  %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %result = torch.aten.amax %input, %dims, %false : !torch.vtensor<[4,8],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4],f32>
  return %result : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: util.func public @scatter_amin$async(
// NONFINITE: arith.constant 0x7C00 : f16
// FINITE: arith.constant 6.550400e+04 : f16
func.func @scatter_amin(%input: !torch.vtensor<[8],f16>, %indices: !torch.vtensor<[4],si64>, %updates: !torch.vtensor<[4],f16>) -> !torch.vtensor<[8],f16> {
  %dim = torch.constant.int 0
  %reduce = torch.constant.str "amin"
  %false = torch.constant.bool false
  %result = torch.aten.scatter_reduce.two %input, %dim, %indices, %updates, %reduce, %false : !torch.vtensor<[8],f16>, !torch.int, !torch.vtensor<[4],si64>, !torch.vtensor<[4],f16>, !torch.str, !torch.bool -> !torch.vtensor<[8],f16>
  return %result : !torch.vtensor<[8],f16>
}
