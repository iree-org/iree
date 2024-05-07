// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-export-parameters{path="opt=%t.irpa" minimum-size=0},iree-io-import-parameters{paths="opt=%t.irpa"})" %s | FileCheck %s

// NOTE: packed types not supported for import yet.
// CHECK: util.global private @constant_scalar_i1 = #flow.parameter.named
util.global private @constant_scalar_i1 = dense<true> : tensor<i1>

// NOTE: packed types not supported for import yet.
// CHECK: util.global private @constant_dense_2xi1 = #flow.parameter.named
util.global private @constant_dense_2xi1 = dense<[true, false]> : tensor<2xi1>

// NOTE: packed types not supported for import yet.
// CHECK: util.global private @constant_dense_3xi4 = #flow.parameter.named
util.global private @constant_dense_3xi4 = dense<[4, 5, 6]> : tensor<3xi4>

// CHECK: util.global private @constant_dense_2xi8 = dense<[4, 5]> : tensor<2xi8>
util.global private @constant_dense_2xi8 = dense<[4, 5]> : tensor<2xi8>

// CHECK: util.global private @constant_dense_2xf32 = dense<[1.100000e+01, 1.200000e+01]> : tensor<2xf32>
util.global private @constant_dense_2xf32 = dense<[1.100000e+01, 1.200000e+01]> : tensor<2xf32>

// CHECK: util.global private @constant_splat_2xf32 = dense<1.100000e+01> : tensor<2xf32>
util.global private @constant_splat_2xf32 = dense<1.100000e+01> : tensor<2xf32>

// NOTE: packed types not supported for import yet.
// CHECK: util.global private mutable @mutable_scalar_i1 = #flow.parameter.named
util.global private mutable @mutable_scalar_i1 = dense<true> : tensor<i1>

// NOTE: packed types not supported for import yet.
// CHECK: util.global private mutable @mutable_dense_3xi4 = #flow.parameter.named
util.global private mutable @mutable_dense_3xi4 = dense<[4, 5, 6]> : tensor<3xi4>

// CHECK: util.global private mutable @mutable_dense_2xf32 = dense<[1.100000e+01, 1.200000e+01]> : tensor<2xf32>
util.global private mutable @mutable_dense_2xf32 = dense<[1.100000e+01, 1.200000e+01]> : tensor<2xf32>

// CHECK: util.global private mutable @mutable_splat_2xf32 = dense<1.100000e+01> : tensor<2xf32>
util.global private mutable @mutable_splat_2xf32 = dense<1.100000e+01> : tensor<2xf32>
