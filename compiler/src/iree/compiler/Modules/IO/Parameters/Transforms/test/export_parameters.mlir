// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-export-parameters{path="opt=%t.irpa" minimum-size=0})" %s | FileCheck %s
// RUN: iree-dump-parameters --parameters=%t.irpa | FileCheck %s --check-prefix=DUMP

//      CHECK: util.global private @constant_scalar_i1 = #flow.parameter.named<"opt"::"constant_scalar_i1"> : tensor<i1>
//       DUMP: - | - | 1 | `constant_scalar_i1`
util.global private @constant_scalar_i1 = dense<true> : tensor<i1>

// CHECK-NEXT: util.global private @constant_dense_2xi1 = #flow.parameter.named<"opt"::"constant_dense_2xi1"> : tensor<2xi1>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 2 | `constant_dense_2xi1`
util.global private @constant_dense_2xi1 = dense<[true, false]> : tensor<2xi1>

// CHECK-NEXT: util.global private @constant_dense_3xi4 = #flow.parameter.named<"opt"::"constant_dense_3xi4"> : tensor<3xi4>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 2 | `constant_dense_3xi4`
util.global private @constant_dense_3xi4 = dense<[4, 5, 6]> : tensor<3xi4>

// CHECK-NEXT: util.global private @constant_dense_2xi8 = #flow.parameter.named<"opt"::"constant_dense_2xi8"> : tensor<2xi8>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 2 | `constant_dense_2xi8`
util.global private @constant_dense_2xi8 = dense<[4, 5]> : tensor<2xi8>

// CHECK-NEXT: util.global private @constant_dense_2xf32 = #flow.parameter.named<"opt"::"constant_dense_2xf32"> : tensor<2xf32>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 8 | `constant_dense_2xf32`
util.global private @constant_dense_2xf32 = dense<[11.0, 12.0]> : tensor<2xf32>

// CHECK-NEXT: util.global private @constant_splat_2xf32 = #flow.parameter.named<"opt"::"constant_splat_2xf32"> : tensor<2xf32>
//  DUMP-NEXT: - | - | 8 | `constant_splat_2xf32`
util.global private @constant_splat_2xf32 = dense<11.0> : tensor<2xf32>

// CHECK-NEXT: util.global private mutable @mutable_scalar_i1 = #flow.parameter.named<"opt"::"mutable_scalar_i1"> : tensor<i1>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 1 | `mutable_scalar_i1`
util.global private mutable @mutable_scalar_i1 = dense<true> : tensor<i1>

// CHECK-NEXT: util.global private mutable @mutable_dense_3xi4 = #flow.parameter.named<"opt"::"mutable_dense_3xi4"> : tensor<3xi4>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 2 | `mutable_dense_3xi4`
util.global private mutable @mutable_dense_3xi4 = dense<[4, 5, 6]> : tensor<3xi4>

// CHECK-NEXT: util.global private mutable @mutable_dense_2xf32 = #flow.parameter.named<"opt"::"mutable_dense_2xf32"> : tensor<2xf32>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 8 | `mutable_dense_2xf32`
util.global private mutable @mutable_dense_2xf32 = dense<[11.0, 12.0]> : tensor<2xf32>

// CHECK-NEXT: util.global private mutable @mutable_splat_2xf32 = #flow.parameter.named<"opt"::"mutable_splat_2xf32"> : tensor<2xf32>
//  DUMP-NEXT: {{[0-9]+}} | {{[0-9]+}} | 8 | `mutable_splat_2xf32`
util.global private mutable @mutable_splat_2xf32 = dense<11.0> : tensor<2xf32>
