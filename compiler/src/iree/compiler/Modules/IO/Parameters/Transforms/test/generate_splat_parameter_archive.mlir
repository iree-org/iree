// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-generate-splat-parameter-archive{file="%t.irpa"})" %s | FileCheck %s
// RUN: iree-dump-parameters --parameters=%t.irpa | FileCheck %s --check-prefix=DUMP

//      CHECK: util.global private @tensor_i1
//       DUMP: - | - | 1 | `tensor_i1`
util.global private @tensor_i1 = #flow.parameter.named<"opt"::"tensor_i1"> : tensor<i1>

// CHECK-NEXT: util.global private @tensor_i8
//  DUMP-NEXT: - | - | 1 | `tensor_i8`
util.global private @tensor_i8 = #flow.parameter.named<"opt"::"tensor_i8"> : tensor<i8>

// CHECK-NEXT: util.global private @tensor_1x2xi32
//  DUMP-NEXT: - | - | 8 | `tensor_1x2xi32`
util.global private @tensor_1x2xi32 = #flow.parameter.named<"opt"::"tensor_1x2xi32"> : tensor<1x2xi32>

// CHECK-NEXT: util.global private @tensor_2x2xi4
//  DUMP-NEXT: - | - | 2 | `tensor_2x2xi4`
util.global private @tensor_2x2xi4 = #flow.parameter.named<"opt"::"tensor_2x2xi4"> : tensor<2x2xi4>

// CHECK-NEXT: util.global private @tensor_3xi4
//  DUMP-NEXT: - | - | 2 | `tensor_3xi4`
util.global private @tensor_3xi4 = #flow.parameter.named<"opt"::"tensor_3xi4"> : tensor<3xi4>

util.func private @function() {
  //      CHECK: flow.tensor.constant
  //  DUMP-NEXT: - | - | 4 | `inline`
  flow.tensor.constant #flow.parameter.named<"opt"::"inline"> : tensor<4xi8>
  util.return
}
