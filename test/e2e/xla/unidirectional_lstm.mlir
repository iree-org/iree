// An example LSTM exported from a python reference model with dummy weights.

// RUN: iree-run-mlir %s --target_backends=interpreter-bytecode --input_values="1x5xf32=[0 1 0 3 4]\n1x5x2x2xf32=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]" --noexport_all --noprint_mlir | FileCheck %s --implicit-check-not="[" --implicit-check-not="]" --dump-input=fail

// Exported via the XLA HLO Importer
// The resulting MLIR was modified by hand by changing all large constants to be
// splats of 0.42, removing the leading "module" wrapper, removing "name"
// attributes, removing extraneous 0s from float constants, and cleaning up
// extra whitespace.

func @Min_reduction.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = xla_hlo.min %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}
func @Max_reduction.51(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @Max_1_reduction.55(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @ForwardLoopBody_o0Jnom3Cdxo__.59(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %cst = constant dense<1> : tensor<i64>
  %1 = xla_hlo.add %0, %cst : tensor<i64>
  %2 = "xla_hlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %3 = "xla_hlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<40xf32>
  %4 = "xla_hlo.get_tuple_element"(%arg0) {index = 3 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %5 = "xla_hlo.get_tuple_element"(%arg0) {index = 4 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<74x40xf32>
  %6 = "xla_hlo.get_tuple_element"(%arg0) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %7 = "xla_hlo.get_tuple_element"(%arg0) {index = 9 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
  %8 = "xla_hlo.gather"(%7, %0) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<1> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x1xf32>, tensor<i64>) -> tensor<1x1xf32>
  %9 = "xla_hlo.reshape"(%8) : (tensor<1x1xf32>) -> tensor<1xf32>
  %10 = "xla_hlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %cst_0 = constant dense<1.0> : tensor<f32>
  %11 = "xla_hlo.broadcast_in_dim"(%cst_0) : (tensor<f32>) -> tensor<1x10xf32>
  %12 = xla_hlo.mul %10, %11 : tensor<1x10xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %13 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<1x10xf32>
  %14 = "xla_hlo.compare"(%12, %13) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %15 = "xla_hlo.get_tuple_element"(%arg0) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %cst_2 = constant dense<5.0e-01> : tensor<f32>
  %16 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %17 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %18 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %19 = "xla_hlo.get_tuple_element"(%arg0) {index = 8 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x64xf32>
  %20 = "xla_hlo.gather"(%19, %0) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<[1, 1, 64]> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x64xf32>, tensor<i64>) -> tensor<1x64xf32>
  %21 = "xla_hlo.get_tuple_element"(%arg0) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %22 = "xla_hlo.concatenate"(%20, %21) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
  %23 = "xla_hlo.dot"(%22, %5) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x74xf32>, tensor<74x40xf32>) -> tensor<1x40xf32>
  %24 = "xla_hlo.transpose"(%23) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x40xf32>
  %25 = "xla_hlo.reshape"(%3) : (tensor<40xf32>) -> tensor<1x40xf32>
  %26 = xla_hlo.add %24, %25 : tensor<1x40xf32>
  %27 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 30]> : tensor<2xi64>, start_indices = dense<[0, 20]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %28 = xla_hlo.mul %18, %27 : tensor<1x10xf32>
  %29 = "xla_hlo.tanh"(%28) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %30 = xla_hlo.mul %17, %29 : tensor<1x10xf32>
  %31 = xla_hlo.add %16, %30 : tensor<1x10xf32>
  %32 = xla_hlo.mul %31, %15 : tensor<1x10xf32>
  %cst_3 = constant dense<5.0e-01> : tensor<f32>
  %33 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %34 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %35 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %36 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 20]> : tensor<2xi64>, start_indices = dense<[0, 10]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %37 = xla_hlo.mul %35, %36 : tensor<1x10xf32>
  %38 = "xla_hlo.tanh"(%37) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %39 = xla_hlo.mul %34, %38 : tensor<1x10xf32>
  %40 = xla_hlo.add %33, %39 : tensor<1x10xf32>
  %41 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %42 = "xla_hlo.tanh"(%41) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %43 = xla_hlo.mul %40, %42 : tensor<1x10xf32>
  %44 = xla_hlo.add %32, %43 : tensor<1x10xf32>
  %cst_4 = constant dense<1.0e+01> : tensor<f32>
  %45 = "xla_hlo.broadcast_in_dim"(%cst_4) : (tensor<f32>) -> tensor<1x10xf32>
  %46 = xla_hlo.min %44, %45 : tensor<1x10xf32>
  %cst_5 = constant dense<-1.0e+01> : tensor<f32>
  %47 = "xla_hlo.broadcast_in_dim"(%cst_5) : (tensor<f32>) -> tensor<1x10xf32>
  %48 = xla_hlo.max %46, %47 : tensor<1x10xf32>
  %49 = "xla_hlo.select"(%14, %15, %48) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %50 = "xla_hlo.reshape"(%8) : (tensor<1x1xf32>) -> tensor<1xf32>
  %51 = "xla_hlo.broadcast_in_dim"(%50) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %cst_6 = constant dense<1.0> : tensor<f32>
  %52 = "xla_hlo.broadcast_in_dim"(%cst_6) : (tensor<f32>) -> tensor<1x10xf32>
  %53 = xla_hlo.mul %51, %52 : tensor<1x10xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %54 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<1x10xf32>
  %55 = "xla_hlo.compare"(%53, %54) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %cst_8 = constant dense<5.0e-01> : tensor<f32>
  %56 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %57 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %58 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %59 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 40]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %60 = xla_hlo.mul %58, %59 : tensor<1x10xf32>
  %61 = "xla_hlo.tanh"(%60) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %62 = xla_hlo.mul %57, %61 : tensor<1x10xf32>
  %63 = xla_hlo.add %56, %62 : tensor<1x10xf32>
  %64 = "xla_hlo.tanh"(%48) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %65 = xla_hlo.mul %63, %64 : tensor<1x10xf32>
  %66 = "xla_hlo.select"(%55, %21, %65) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %67 = "xla_hlo.get_tuple_element"(%arg0) {index = 10 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
  %68 = "xla_hlo.get_tuple_element"(%arg0) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %69 = "xla_hlo.reshape"(%6) : (tensor<i64>) -> tensor<1xi64>
  %70 = "xla_hlo.slice"(%69) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
  %71 = "xla_hlo.reshape"(%0) : (tensor<i64>) -> tensor<1xi64>
  %72 = "xla_hlo.concatenate"(%71) {dimension = 0 : i64} : (tensor<1xi64>) -> tensor<1xi64>
  %73 = "xla_hlo.convert"(%72) : (tensor<1xi64>) -> tensor<1xi32>
  %74 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %75 = "xla_hlo.reshape"(%74) : (tensor<1xi32>) -> tensor<i32>
  %76 = "xla_hlo.dynamic-update-slice"(%68, %70, %75) : (tensor<5xi64>, tensor<1xi64>, tensor<i32>) -> tensor<5xi64>
  %77 = "xla_hlo.get_tuple_element"(%arg0) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %78 = "xla_hlo.reshape"(%49) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %79 = "xla_hlo.slice"(%78) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
  %80 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %81 = "xla_hlo.reshape"(%80) : (tensor<1xi32>) -> tensor<i32>
  %cst_9 = constant dense<0> : tensor<i32>
  %82 = "xla_hlo.dynamic-update-slice"(%77, %79, %81, %cst_9, %cst_9) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %83 = "xla_hlo.get_tuple_element"(%arg0) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %84 = "xla_hlo.reshape"(%66) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %85 = "xla_hlo.slice"(%84) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
  %86 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %87 = "xla_hlo.reshape"(%86) : (tensor<1xi32>) -> tensor<i32>
  %cst_10 = constant dense<0> : tensor<i32>
  %88 = "xla_hlo.dynamic-update-slice"(%83, %85, %87, %cst_10, %cst_10) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %89 = "xla_hlo.tuple"(%1, %2, %3, %4, %5, %6, %49, %66, %19, %7, %67, %76, %82, %88) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  return %89 : tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
}
func @ForwardLoopCond_gFAnjWGSoLs__.167(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %1 = "xla_hlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %2 = "xla_hlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %3 = "xla_hlo.tuple"(%2) : (tensor<i1>) -> tuple<tensor<i1>>
  return %3 : tuple<tensor<i1>>
}
func @cond_wrapper.185(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i1> {
  %0 = call @ForwardLoopCond_gFAnjWGSoLs__.167(%arg0) : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>>
  %1 = "xla_hlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<i1>>) -> tensor<i1>
  return %1 : tensor<i1>
}
func @Forward_DIZIAkooG44__disable_call_shape_inference_true_.189(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<5x1x64xf32>, %arg3: tensor<5x1x1xf32>, %arg4: tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>> {
  %cst = constant dense<5> : tensor<i32>
  %0 = "xla_hlo.convert"(%arg3) : (tensor<5x1x1xf32>) -> tensor<5x1x1xf32>
  %cst_0 = constant dense<0x7F800000> : tensor<f32>
  %1 = "xla_hlo.convert"(%cst_0) : (tensor<f32>) -> tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):  // no predecessors
  %42 = xla_hlo.min %arg5, %arg6 : tensor<f32>
  "xla_hlo.return"(%42) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  %3 = "xla_hlo.convert"(%2) : (tensor<5xf32>) -> tensor<5xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %4 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<5xf32>
  %5 = "xla_hlo.compare"(%3, %4) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %6 = "xla_hlo.convert"(%5) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_2 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %7 = xla_hlo.mul %6, %cst_2 : tensor<5xi32>
  %8 = "xla_hlo.convert"(%7) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_3 = constant dense<-2147483648> : tensor<i32>
  %9 = "xla_hlo.convert"(%cst_3) : (tensor<i32>) -> tensor<i32>
  %10 = "xla_hlo.reduce"(%8, %9) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
  %42 = xla_hlo.max %arg5, %arg6 : tensor<i32>
  "xla_hlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %11 = "xla_hlo.convert"(%10) : (tensor<i32>) -> tensor<i32>
  %12 = xla_hlo.sub %cst, %11 : tensor<i32>
  %cst_4 = constant dense<5> : tensor<i32>
  %13 = "xla_hlo.compare"(%12, %cst_4) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %cst_5 = constant dense<0> : tensor<i32>
  %cst_6 = constant dense<5> : tensor<i32>
  %14 = "xla_hlo.reverse"(%3) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %15 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<5xf32>
  %16 = "xla_hlo.compare"(%14, %15) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %17 = "xla_hlo.convert"(%16) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_8 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %18 = xla_hlo.mul %17, %cst_8 : tensor<5xi32>
  %19 = "xla_hlo.convert"(%18) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_9 = constant dense<-2147483648> : tensor<i32>
  %20 = "xla_hlo.convert"(%cst_9) : (tensor<i32>) -> tensor<i32>
  %21 = "xla_hlo.reduce"(%19, %20) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
  %42 = xla_hlo.max %arg5, %arg6 : tensor<i32>
  "xla_hlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %22 = "xla_hlo.convert"(%21) : (tensor<i32>) -> tensor<i32>
  %23 = xla_hlo.sub %cst_6, %22 : tensor<i32>
  %24 = "xla_hlo.select"(%13, %cst_5, %23) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %25 = "xla_hlo.convert"(%24) : (tensor<i32>) -> tensor<i64>
  %cst_10 = constant dense<5> : tensor<i32>
  %26 = xla_hlo.sub %cst_10, %12 : tensor<i32>
  %27 = "xla_hlo.convert"(%26) : (tensor<i32>) -> tensor<i64>
  %cst_11 = constant dense<0.0> : tensor<f32>
  %28 = "xla_hlo.broadcast_in_dim"(%cst_11) : (tensor<f32>) -> tensor<40xf32>
  %cst_12 = constant dense<0> : tensor<i64>
  %cst_13 = constant dense<0.42> : tensor<74x40xf32>
  %cst_14 = constant dense<0> : tensor<i64>
  %cst_15 = constant dense<0> : tensor<i64>
  %29 = "xla_hlo.broadcast_in_dim"(%cst_15) : (tensor<i64>) -> tensor<5xi64>
  %cst_16 = constant dense<0.0> : tensor<f32>
  %30 = "xla_hlo.broadcast_in_dim"(%cst_16) : (tensor<f32>) -> tensor<5x1x10xf32>
  %cst_17 = constant dense<0.0> : tensor<f32>
  %31 = "xla_hlo.broadcast_in_dim"(%cst_17) : (tensor<f32>) -> tensor<5x1x10xf32>
  %32 = "xla_hlo.tuple"(%25, %27, %28, %cst_12, %cst_13, %cst_14, %arg0, %arg1, %arg2, %arg3, %arg4, %29, %30, %31) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %33 = "xla_hlo.while"(%32) {body = @ForwardLoopBody_o0Jnom3Cdxo__.59, cond = @cond_wrapper.185} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %34 = "xla_hlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %35 = "xla_hlo.get_tuple_element"(%33) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %36 = "xla_hlo.get_tuple_element"(%33) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %37 = "xla_hlo.get_tuple_element"(%33) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %38 = "xla_hlo.get_tuple_element"(%33) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %39 = "xla_hlo.get_tuple_element"(%33) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %40 = "xla_hlo.get_tuple_element"(%33) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %41 = "xla_hlo.tuple"(%34, %35, %36, %37, %38, %39, %40) : (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  return %41 : tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
}
func @Min_reduction.316(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = xla_hlo.min %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}
func @Max_reduction.320(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @Max_1_reduction.324(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @ForwardLoopBody_o0Jnom3Cdxo__.328(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %cst = constant dense<1> : tensor<i64>
  %1 = xla_hlo.add %0, %cst : tensor<i64>
  %2 = "xla_hlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %3 = "xla_hlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<40xf32>
  %4 = "xla_hlo.get_tuple_element"(%arg0) {index = 3 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %5 = "xla_hlo.get_tuple_element"(%arg0) {index = 4 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<20x40xf32>
  %6 = "xla_hlo.get_tuple_element"(%arg0) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %7 = "xla_hlo.get_tuple_element"(%arg0) {index = 9 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
  %8 = "xla_hlo.gather"(%7, %0) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<1> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x1xf32>, tensor<i64>) -> tensor<1x1xf32>
  %9 = "xla_hlo.reshape"(%8) : (tensor<1x1xf32>) -> tensor<1xf32>
  %10 = "xla_hlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %cst_0 = constant dense<1.0> : tensor<f32>
  %11 = "xla_hlo.broadcast_in_dim"(%cst_0) : (tensor<f32>) -> tensor<1x10xf32>
  %12 = xla_hlo.mul %10, %11 : tensor<1x10xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %13 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<1x10xf32>
  %14 = "xla_hlo.compare"(%12, %13) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %15 = "xla_hlo.get_tuple_element"(%arg0) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %cst_2 = constant dense<5.0e-01> : tensor<f32>
  %16 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %17 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %18 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %19 = "xla_hlo.get_tuple_element"(%arg0) {index = 8 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %20 = "xla_hlo.gather"(%19, %0) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<[1, 1, 10]> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x10xf32>, tensor<i64>) -> tensor<1x10xf32>
  %21 = "xla_hlo.get_tuple_element"(%arg0) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %22 = "xla_hlo.concatenate"(%20, %21) {dimension = 1 : i64} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x20xf32>
  %23 = "xla_hlo.dot"(%22, %5) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x20xf32>, tensor<20x40xf32>) -> tensor<1x40xf32>
  %24 = "xla_hlo.transpose"(%23) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x40xf32>
  %25 = "xla_hlo.reshape"(%3) : (tensor<40xf32>) -> tensor<1x40xf32>
  %26 = xla_hlo.add %24, %25 : tensor<1x40xf32>
  %27 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 30]> : tensor<2xi64>, start_indices = dense<[0, 20]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %28 = xla_hlo.mul %18, %27 : tensor<1x10xf32>
  %29 = "xla_hlo.tanh"(%28) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %30 = xla_hlo.mul %17, %29 : tensor<1x10xf32>
  %31 = xla_hlo.add %16, %30 : tensor<1x10xf32>
  %32 = xla_hlo.mul %31, %15 : tensor<1x10xf32>
  %cst_3 = constant dense<5.0e-01> : tensor<f32>
  %33 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %34 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %35 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %36 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 20]> : tensor<2xi64>, start_indices = dense<[0, 10]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %37 = xla_hlo.mul %35, %36 : tensor<1x10xf32>
  %38 = "xla_hlo.tanh"(%37) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %39 = xla_hlo.mul %34, %38 : tensor<1x10xf32>
  %40 = xla_hlo.add %33, %39 : tensor<1x10xf32>
  %41 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %42 = "xla_hlo.tanh"(%41) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %43 = xla_hlo.mul %40, %42 : tensor<1x10xf32>
  %44 = xla_hlo.add %32, %43 : tensor<1x10xf32>
  %cst_4 = constant dense<1.0e+01> : tensor<f32>
  %45 = "xla_hlo.broadcast_in_dim"(%cst_4) : (tensor<f32>) -> tensor<1x10xf32>
  %46 = xla_hlo.min %44, %45 : tensor<1x10xf32>
  %cst_5 = constant dense<-1.0e+01> : tensor<f32>
  %47 = "xla_hlo.broadcast_in_dim"(%cst_5) : (tensor<f32>) -> tensor<1x10xf32>
  %48 = xla_hlo.max %46, %47 : tensor<1x10xf32>
  %49 = "xla_hlo.select"(%14, %15, %48) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %50 = "xla_hlo.reshape"(%8) : (tensor<1x1xf32>) -> tensor<1xf32>
  %51 = "xla_hlo.broadcast_in_dim"(%50) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %cst_6 = constant dense<1.0> : tensor<f32>
  %52 = "xla_hlo.broadcast_in_dim"(%cst_6) : (tensor<f32>) -> tensor<1x10xf32>
  %53 = xla_hlo.mul %51, %52 : tensor<1x10xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %54 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<1x10xf32>
  %55 = "xla_hlo.compare"(%53, %54) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %cst_8 = constant dense<5.0e-01> : tensor<f32>
  %56 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %57 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %58 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %59 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 40]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %60 = xla_hlo.mul %58, %59 : tensor<1x10xf32>
  %61 = "xla_hlo.tanh"(%60) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %62 = xla_hlo.mul %57, %61 : tensor<1x10xf32>
  %63 = xla_hlo.add %56, %62 : tensor<1x10xf32>
  %64 = "xla_hlo.tanh"(%48) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %65 = xla_hlo.mul %63, %64 : tensor<1x10xf32>
  %66 = "xla_hlo.select"(%55, %21, %65) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %67 = "xla_hlo.get_tuple_element"(%arg0) {index = 10 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
  %68 = "xla_hlo.get_tuple_element"(%arg0) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %69 = "xla_hlo.reshape"(%6) : (tensor<i64>) -> tensor<1xi64>
  %70 = "xla_hlo.slice"(%69) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
  %71 = "xla_hlo.reshape"(%0) : (tensor<i64>) -> tensor<1xi64>
  %72 = "xla_hlo.concatenate"(%71) {dimension = 0 : i64} : (tensor<1xi64>) -> tensor<1xi64>
  %73 = "xla_hlo.convert"(%72) : (tensor<1xi64>) -> tensor<1xi32>
  %74 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %75 = "xla_hlo.reshape"(%74) : (tensor<1xi32>) -> tensor<i32>
  %76 = "xla_hlo.dynamic-update-slice"(%68, %70, %75) : (tensor<5xi64>, tensor<1xi64>, tensor<i32>) -> tensor<5xi64>
  %77 = "xla_hlo.get_tuple_element"(%arg0) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %78 = "xla_hlo.reshape"(%49) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %79 = "xla_hlo.slice"(%78) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
  %80 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %81 = "xla_hlo.reshape"(%80) : (tensor<1xi32>) -> tensor<i32>
  %cst_9 = constant dense<0> : tensor<i32>
  %82 = "xla_hlo.dynamic-update-slice"(%77, %79, %81, %cst_9, %cst_9) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %83 = "xla_hlo.get_tuple_element"(%arg0) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %84 = "xla_hlo.reshape"(%66) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %85 = "xla_hlo.slice"(%84) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
  %86 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %87 = "xla_hlo.reshape"(%86) : (tensor<1xi32>) -> tensor<i32>
  %cst_10 = constant dense<0> : tensor<i32>
  %88 = "xla_hlo.dynamic-update-slice"(%83, %85, %87, %cst_10, %cst_10) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %89 = "xla_hlo.tuple"(%1, %2, %3, %4, %5, %6, %49, %66, %19, %7, %67, %76, %82, %88) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  return %89 : tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
}
func @ForwardLoopCond_gFAnjWGSoLs__.436(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %1 = "xla_hlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %2 = "xla_hlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %3 = "xla_hlo.tuple"(%2) : (tensor<i1>) -> tuple<tensor<i1>>
  return %3 : tuple<tensor<i1>>
}
func @cond_wrapper.454(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i1> {
  %0 = call @ForwardLoopCond_gFAnjWGSoLs__.436(%arg0) : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>>
  %1 = "xla_hlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<i1>>) -> tensor<i1>
  return %1 : tensor<i1>
}
func @Forward_DIZIAkooG44__disable_call_shape_inference_true_.458(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<5x1x10xf32>, %arg3: tensor<5x1x1xf32>, %arg4: tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>> {
  %cst = constant dense<5> : tensor<i32>
  %0 = "xla_hlo.convert"(%arg3) : (tensor<5x1x1xf32>) -> tensor<5x1x1xf32>
  %cst_0 = constant dense<0x7F800000> : tensor<f32>
  %1 = "xla_hlo.convert"(%cst_0) : (tensor<f32>) -> tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):  // no predecessors
  %42 = xla_hlo.min %arg5, %arg6 : tensor<f32>
  "xla_hlo.return"(%42) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  %3 = "xla_hlo.convert"(%2) : (tensor<5xf32>) -> tensor<5xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %4 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<5xf32>
  %5 = "xla_hlo.compare"(%3, %4) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %6 = "xla_hlo.convert"(%5) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_2 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %7 = xla_hlo.mul %6, %cst_2 : tensor<5xi32>
  %8 = "xla_hlo.convert"(%7) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_3 = constant dense<-2147483648> : tensor<i32>
  %9 = "xla_hlo.convert"(%cst_3) : (tensor<i32>) -> tensor<i32>
  %10 = "xla_hlo.reduce"(%8, %9) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
  %42 = xla_hlo.max %arg5, %arg6 : tensor<i32>
  "xla_hlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %11 = "xla_hlo.convert"(%10) : (tensor<i32>) -> tensor<i32>
  %12 = xla_hlo.sub %cst, %11 : tensor<i32>
  %cst_4 = constant dense<5> : tensor<i32>
  %13 = "xla_hlo.compare"(%12, %cst_4) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %cst_5 = constant dense<0> : tensor<i32>
  %cst_6 = constant dense<5> : tensor<i32>
  %14 = "xla_hlo.reverse"(%3) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %15 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<5xf32>
  %16 = "xla_hlo.compare"(%14, %15) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %17 = "xla_hlo.convert"(%16) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_8 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %18 = xla_hlo.mul %17, %cst_8 : tensor<5xi32>
  %19 = "xla_hlo.convert"(%18) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_9 = constant dense<-2147483648> : tensor<i32>
  %20 = "xla_hlo.convert"(%cst_9) : (tensor<i32>) -> tensor<i32>
  %21 = "xla_hlo.reduce"(%19, %20) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
  %42 = xla_hlo.max %arg5, %arg6 : tensor<i32>
  "xla_hlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %22 = "xla_hlo.convert"(%21) : (tensor<i32>) -> tensor<i32>
  %23 = xla_hlo.sub %cst_6, %22 : tensor<i32>
  %24 = "xla_hlo.select"(%13, %cst_5, %23) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %25 = "xla_hlo.convert"(%24) : (tensor<i32>) -> tensor<i64>
  %cst_10 = constant dense<5> : tensor<i32>
  %26 = xla_hlo.sub %cst_10, %12 : tensor<i32>
  %27 = "xla_hlo.convert"(%26) : (tensor<i32>) -> tensor<i64>
  %cst_11 = constant dense<0.0> : tensor<f32>
  %28 = "xla_hlo.broadcast_in_dim"(%cst_11) : (tensor<f32>) -> tensor<40xf32>
  %cst_12 = constant dense<0> : tensor<i64>
  %cst_13 = constant dense<0.42> : tensor<20x40xf32>
  %cst_14 = constant dense<-2130569576> : tensor<i64>
  %cst_15 = constant dense<0> : tensor<i64>
  %29 = "xla_hlo.broadcast_in_dim"(%cst_15) : (tensor<i64>) -> tensor<5xi64>
  %cst_16 = constant dense<0.0> : tensor<f32>
  %30 = "xla_hlo.broadcast_in_dim"(%cst_16) : (tensor<f32>) -> tensor<5x1x10xf32>
  %cst_17 = constant dense<0.0> : tensor<f32>
  %31 = "xla_hlo.broadcast_in_dim"(%cst_17) : (tensor<f32>) -> tensor<5x1x10xf32>
  %32 = "xla_hlo.tuple"(%25, %27, %28, %cst_12, %cst_13, %cst_14, %arg0, %arg1, %arg2, %arg3, %arg4, %29, %30, %31) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %33 = "xla_hlo.while"(%32) {body = @ForwardLoopBody_o0Jnom3Cdxo__.328, cond = @cond_wrapper.454} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %34 = "xla_hlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %35 = "xla_hlo.get_tuple_element"(%33) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %36 = "xla_hlo.get_tuple_element"(%33) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %37 = "xla_hlo.get_tuple_element"(%33) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %38 = "xla_hlo.get_tuple_element"(%33) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %39 = "xla_hlo.get_tuple_element"(%33) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %40 = "xla_hlo.get_tuple_element"(%33) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %41 = "xla_hlo.tuple"(%34, %35, %36, %37, %38, %39, %40) : (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  return %41 : tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
}
func @Min_reduction.585(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = xla_hlo.min %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}
func @Max_reduction.589(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @Max_1_reduction.593(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = xla_hlo.max %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @ForwardLoopBody_o0Jnom3Cdxo__.597(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %cst = constant dense<1> : tensor<i64>
  %1 = xla_hlo.add %0, %cst : tensor<i64>
  %2 = "xla_hlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %3 = "xla_hlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<40xf32>
  %4 = "xla_hlo.get_tuple_element"(%arg0) {index = 3 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %5 = "xla_hlo.get_tuple_element"(%arg0) {index = 4 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<20x40xf32>
  %6 = "xla_hlo.get_tuple_element"(%arg0) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %7 = "xla_hlo.get_tuple_element"(%arg0) {index = 9 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
  %8 = "xla_hlo.gather"(%7, %0) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<1> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x1xf32>, tensor<i64>) -> tensor<1x1xf32>
  %9 = "xla_hlo.reshape"(%8) : (tensor<1x1xf32>) -> tensor<1xf32>
  %10 = "xla_hlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %cst_0 = constant dense<1.0> : tensor<f32>
  %11 = "xla_hlo.broadcast_in_dim"(%cst_0) : (tensor<f32>) -> tensor<1x10xf32>
  %12 = xla_hlo.mul %10, %11 : tensor<1x10xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %13 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<1x10xf32>
  %14 = "xla_hlo.compare"(%12, %13) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %15 = "xla_hlo.get_tuple_element"(%arg0) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %cst_2 = constant dense<5.0e-01> : tensor<f32>
  %16 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %17 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %18 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %19 = "xla_hlo.get_tuple_element"(%arg0) {index = 8 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %20 = "xla_hlo.gather"(%19, %0) {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, slice_sizes = dense<[1, 1, 10]> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x10xf32>, tensor<i64>) -> tensor<1x10xf32>
  %21 = "xla_hlo.get_tuple_element"(%arg0) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %22 = "xla_hlo.concatenate"(%20, %21) {dimension = 1 : i64} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x20xf32>
  %23 = "xla_hlo.dot"(%22, %5) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x20xf32>, tensor<20x40xf32>) -> tensor<1x40xf32>
  %24 = "xla_hlo.transpose"(%23) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x40xf32>
  %25 = "xla_hlo.reshape"(%3) : (tensor<40xf32>) -> tensor<1x40xf32>
  %26 = xla_hlo.add %24, %25 : tensor<1x40xf32>
  %27 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 30]> : tensor<2xi64>, start_indices = dense<[0, 20]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %28 = xla_hlo.mul %18, %27 : tensor<1x10xf32>
  %29 = "xla_hlo.tanh"(%28) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %30 = xla_hlo.mul %17, %29 : tensor<1x10xf32>
  %31 = xla_hlo.add %16, %30 : tensor<1x10xf32>
  %32 = xla_hlo.mul %31, %15 : tensor<1x10xf32>
  %cst_3 = constant dense<5.0e-01> : tensor<f32>
  %33 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %34 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %35 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %36 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 20]> : tensor<2xi64>, start_indices = dense<[0, 10]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %37 = xla_hlo.mul %35, %36 : tensor<1x10xf32>
  %38 = "xla_hlo.tanh"(%37) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %39 = xla_hlo.mul %34, %38 : tensor<1x10xf32>
  %40 = xla_hlo.add %33, %39 : tensor<1x10xf32>
  %41 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %42 = "xla_hlo.tanh"(%41) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %43 = xla_hlo.mul %40, %42 : tensor<1x10xf32>
  %44 = xla_hlo.add %32, %43 : tensor<1x10xf32>
  %cst_4 = constant dense<1.0e+01> : tensor<f32>
  %45 = "xla_hlo.broadcast_in_dim"(%cst_4) : (tensor<f32>) -> tensor<1x10xf32>
  %46 = xla_hlo.min %44, %45 : tensor<1x10xf32>
  %cst_5 = constant dense<-1.0e+01> : tensor<f32>
  %47 = "xla_hlo.broadcast_in_dim"(%cst_5) : (tensor<f32>) -> tensor<1x10xf32>
  %48 = xla_hlo.max %46, %47 : tensor<1x10xf32>
  %49 = "xla_hlo.select"(%14, %15, %48) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %50 = "xla_hlo.reshape"(%8) : (tensor<1x1xf32>) -> tensor<1xf32>
  %51 = "xla_hlo.broadcast_in_dim"(%50) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %cst_6 = constant dense<1.0> : tensor<f32>
  %52 = "xla_hlo.broadcast_in_dim"(%cst_6) : (tensor<f32>) -> tensor<1x10xf32>
  %53 = xla_hlo.mul %51, %52 : tensor<1x10xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %54 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<1x10xf32>
  %55 = "xla_hlo.compare"(%53, %54) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %cst_8 = constant dense<5.0e-01> : tensor<f32>
  %56 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %57 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %58 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<1x10xf32>
  %59 = "xla_hlo.slice"(%26) {limit_indices = dense<[1, 40]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %60 = xla_hlo.mul %58, %59 : tensor<1x10xf32>
  %61 = "xla_hlo.tanh"(%60) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %62 = xla_hlo.mul %57, %61 : tensor<1x10xf32>
  %63 = xla_hlo.add %56, %62 : tensor<1x10xf32>
  %64 = "xla_hlo.tanh"(%48) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %65 = xla_hlo.mul %63, %64 : tensor<1x10xf32>
  %66 = "xla_hlo.select"(%55, %21, %65) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %67 = "xla_hlo.get_tuple_element"(%arg0) {index = 10 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
  %68 = "xla_hlo.get_tuple_element"(%arg0) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %69 = "xla_hlo.reshape"(%6) : (tensor<i64>) -> tensor<1xi64>
  %70 = "xla_hlo.slice"(%69) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
  %71 = "xla_hlo.reshape"(%0) : (tensor<i64>) -> tensor<1xi64>
  %72 = "xla_hlo.concatenate"(%71) {dimension = 0 : i64} : (tensor<1xi64>) -> tensor<1xi64>
  %73 = "xla_hlo.convert"(%72) : (tensor<1xi64>) -> tensor<1xi32>
  %74 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %75 = "xla_hlo.reshape"(%74) : (tensor<1xi32>) -> tensor<i32>
  %76 = "xla_hlo.dynamic-update-slice"(%68, %70, %75) : (tensor<5xi64>, tensor<1xi64>, tensor<i32>) -> tensor<5xi64>
  %77 = "xla_hlo.get_tuple_element"(%arg0) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %78 = "xla_hlo.reshape"(%49) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %79 = "xla_hlo.slice"(%78) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
  %80 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %81 = "xla_hlo.reshape"(%80) : (tensor<1xi32>) -> tensor<i32>
  %cst_9 = constant dense<0> : tensor<i32>
  %82 = "xla_hlo.dynamic-update-slice"(%77, %79, %81, %cst_9, %cst_9) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %83 = "xla_hlo.get_tuple_element"(%arg0) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %84 = "xla_hlo.reshape"(%66) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %85 = "xla_hlo.slice"(%84) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
  %86 = "xla_hlo.slice"(%73) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  %87 = "xla_hlo.reshape"(%86) : (tensor<1xi32>) -> tensor<i32>
  %cst_10 = constant dense<0> : tensor<i32>
  %88 = "xla_hlo.dynamic-update-slice"(%83, %85, %87, %cst_10, %cst_10) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %89 = "xla_hlo.tuple"(%1, %2, %3, %4, %5, %6, %49, %66, %19, %7, %67, %76, %82, %88) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  return %89 : tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
}
func @ForwardLoopCond_gFAnjWGSoLs__.705(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %1 = "xla_hlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %2 = "xla_hlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %3 = "xla_hlo.tuple"(%2) : (tensor<i1>) -> tuple<tensor<i1>>
  return %3 : tuple<tensor<i1>>
}
func @cond_wrapper.723(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i1> {
  %0 = call @ForwardLoopCond_gFAnjWGSoLs__.705(%arg0) : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>>
  %1 = "xla_hlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<i1>>) -> tensor<i1>
  return %1 : tensor<i1>
}
func @Forward_DIZIAkooG44__disable_call_shape_inference_true_.727(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<5x1x10xf32>, %arg3: tensor<5x1x1xf32>, %arg4: tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>> {
  %cst = constant dense<5> : tensor<i32>
  %0 = "xla_hlo.convert"(%arg3) : (tensor<5x1x1xf32>) -> tensor<5x1x1xf32>
  %cst_0 = constant dense<0x7F800000> : tensor<f32>
  %1 = "xla_hlo.convert"(%cst_0) : (tensor<f32>) -> tensor<f32>
  %2 = "xla_hlo.reduce"(%0, %1) ( {
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):  // no predecessors
  %42 = xla_hlo.min %arg5, %arg6 : tensor<f32>
  "xla_hlo.return"(%42) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  %3 = "xla_hlo.convert"(%2) : (tensor<5xf32>) -> tensor<5xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %4 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<5xf32>
  %5 = "xla_hlo.compare"(%3, %4) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %6 = "xla_hlo.convert"(%5) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_2 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %7 = xla_hlo.mul %6, %cst_2 : tensor<5xi32>
  %8 = "xla_hlo.convert"(%7) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_3 = constant dense<-2147483648> : tensor<i32>
  %9 = "xla_hlo.convert"(%cst_3) : (tensor<i32>) -> tensor<i32>
  %10 = "xla_hlo.reduce"(%8, %9) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
  %42 = xla_hlo.max %arg5, %arg6 : tensor<i32>
  "xla_hlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %11 = "xla_hlo.convert"(%10) : (tensor<i32>) -> tensor<i32>
  %12 = xla_hlo.sub %cst, %11 : tensor<i32>
  %cst_4 = constant dense<5> : tensor<i32>
  %13 = "xla_hlo.compare"(%12, %cst_4) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %cst_5 = constant dense<0> : tensor<i32>
  %cst_6 = constant dense<5> : tensor<i32>
  %14 = "xla_hlo.reverse"(%3) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %15 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<5xf32>
  %16 = "xla_hlo.compare"(%14, %15) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %17 = "xla_hlo.convert"(%16) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_8 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %18 = xla_hlo.mul %17, %cst_8 : tensor<5xi32>
  %19 = "xla_hlo.convert"(%18) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_9 = constant dense<-2147483648> : tensor<i32>
  %20 = "xla_hlo.convert"(%cst_9) : (tensor<i32>) -> tensor<i32>
  %21 = "xla_hlo.reduce"(%19, %20) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
  %42 = xla_hlo.max %arg5, %arg6 : tensor<i32>
  "xla_hlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %22 = "xla_hlo.convert"(%21) : (tensor<i32>) -> tensor<i32>
  %23 = xla_hlo.sub %cst_6, %22 : tensor<i32>
  %24 = "xla_hlo.select"(%13, %cst_5, %23) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %25 = "xla_hlo.convert"(%24) : (tensor<i32>) -> tensor<i64>
  %cst_10 = constant dense<5> : tensor<i32>
  %26 = xla_hlo.sub %cst_10, %12 : tensor<i32>
  %27 = "xla_hlo.convert"(%26) : (tensor<i32>) -> tensor<i64>
  %cst_11 = constant dense<0.0> : tensor<f32>
  %28 = "xla_hlo.broadcast_in_dim"(%cst_11) : (tensor<f32>) -> tensor<40xf32>
  %cst_12 = constant dense<0> : tensor<i64>
  %cst_13 = constant dense<0.42> : tensor<20x40xf32>
  %cst_14 = constant dense<-1356785880> : tensor<i64>
  %cst_15 = constant dense<0> : tensor<i64>
  %29 = "xla_hlo.broadcast_in_dim"(%cst_15) : (tensor<i64>) -> tensor<5xi64>
  %cst_16 = constant dense<0.0> : tensor<f32>
  %30 = "xla_hlo.broadcast_in_dim"(%cst_16) : (tensor<f32>) -> tensor<5x1x10xf32>
  %cst_17 = constant dense<0.0> : tensor<f32>
  %31 = "xla_hlo.broadcast_in_dim"(%cst_17) : (tensor<f32>) -> tensor<5x1x10xf32>
  %32 = "xla_hlo.tuple"(%25, %27, %28, %cst_12, %cst_13, %cst_14, %arg0, %arg1, %arg2, %arg3, %arg4, %29, %30, %31) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %33 = "xla_hlo.while"(%32) {body = @ForwardLoopBody_o0Jnom3Cdxo__.597, cond = @cond_wrapper.723} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %34 = "xla_hlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %35 = "xla_hlo.get_tuple_element"(%33) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %36 = "xla_hlo.get_tuple_element"(%33) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %37 = "xla_hlo.get_tuple_element"(%33) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %38 = "xla_hlo.get_tuple_element"(%33) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %39 = "xla_hlo.get_tuple_element"(%33) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %40 = "xla_hlo.get_tuple_element"(%33) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<20x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %41 = "xla_hlo.tuple"(%34, %35, %36, %37, %38, %39, %40) : (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  return %41 : tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
}

// CHECK-LABEL: EXEC @main
func @main(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5x2x2xf32>) -> tuple<tensor<5x1x10xf32>>
  attributes { iree.module.export } {
  %cst = constant dense<0.0> : tensor<f32>
  %0 = "xla_hlo.broadcast_in_dim"(%cst) : (tensor<f32>) -> tensor<1x10xf32>
  %cst_0 = constant dense<0.0> : tensor<f32>
  %1 = "xla_hlo.broadcast_in_dim"(%cst_0) : (tensor<f32>) -> tensor<1x10xf32>
  %cst_1 = constant dense<0.0> : tensor<f32>
  %2 = "xla_hlo.broadcast_in_dim"(%cst_1) : (tensor<f32>) -> tensor<1x10xf32>
  %cst_2 = constant dense<0.0> : tensor<f32>
  %3 = "xla_hlo.broadcast_in_dim"(%cst_2) : (tensor<f32>) -> tensor<1x10xf32>
  %cst_3 = constant dense<0.0> : tensor<f32>
  %4 = "xla_hlo.broadcast_in_dim"(%cst_3) : (tensor<f32>) -> tensor<1x10xf32>
  %cst_4 = constant dense<0.0> : tensor<f32>
  %5 = "xla_hlo.broadcast_in_dim"(%cst_4) : (tensor<f32>) -> tensor<1x10xf32>
  %6 = "xla_hlo.reshape"(%arg1) : (tensor<1x5x2x2xf32>) -> tensor<1x5x2x2xf32>
  %7 = "xla_hlo.reshape"(%6) : (tensor<1x5x2x2xf32>) -> tensor<1x5x4xf32>
  %cst_5 = constant dense<0.0> : tensor<f32>
  %8 = "xla_hlo.pad"(%7, %cst_5) {edge_padding_high = dense<[0, 0, 60]> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>} : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x5x64xf32>
  %9 = "xla_hlo.transpose"(%8) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<1x5x64xf32>) -> tensor<5x1x64xf32>
  %10 = "xla_hlo.reshape"(%arg0) : (tensor<1x5xf32>) -> tensor<1x5xf32>
  %11 = "xla_hlo.transpose"(%10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x5xf32>) -> tensor<5x1xf32>
  %12 = "xla_hlo.reshape"(%11) : (tensor<5x1xf32>) -> tensor<5x1x1xf32>
  %cst_6 = constant dense<0.0> : tensor<f32>
  %13 = "xla_hlo.broadcast_in_dim"(%cst_6) : (tensor<f32>) -> tensor<5x1x1xf32>
  %14 = call @Forward_DIZIAkooG44__disable_call_shape_inference_true_.189(%4, %5, %9, %12, %13) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  %15 = "xla_hlo.get_tuple_element"(%14) {index = 3 : i32} : (tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>) -> tensor<5x1x10xf32>
  %cst_7 = constant dense<0.0> : tensor<f32>
  %16 = "xla_hlo.broadcast_in_dim"(%cst_7) : (tensor<f32>) -> tensor<5x1x1xf32>
  %17 = call @Forward_DIZIAkooG44__disable_call_shape_inference_true_.458(%2, %3, %15, %12, %16) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  %18 = "xla_hlo.get_tuple_element"(%17) {index = 3 : i32} : (tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>) -> tensor<5x1x10xf32>
  %cst_8 = constant dense<0.0> : tensor<f32>
  %19 = "xla_hlo.broadcast_in_dim"(%cst_8) : (tensor<f32>) -> tensor<5x1x1xf32>
  %20 = call @Forward_DIZIAkooG44__disable_call_shape_inference_true_.727(%0, %1, %18, %12, %19) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x10xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  %21 = "xla_hlo.get_tuple_element"(%20) {index = 3 : i32} : (tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>) -> tensor<5x1x10xf32>
  %22 = "xla_hlo.copy"(%21) : (tensor<5x1x10xf32>) -> tensor<5x1x10xf32>
  %23 = "xla_hlo.reshape"(%22) : (tensor<5x1x10xf32>) -> tensor<5x1x10xf32>
  %24 = "xla_hlo.tuple"(%23) : (tensor<5x1x10xf32>) -> tuple<tensor<5x1x10xf32>>
  return %24 : tuple<tensor<5x1x10xf32>>
}

// CHECK: 5x1x10xf32=
// CHECK-SAME: [
// CHECK-SAME:   [0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}}]
// CHECK-SAME: ][
// CHECK-SAME:   [0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}} 0.7{{[0-9]+}}]
// CHECK-SAME: ][
// CHECK-SAME:   [0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}} 0.9{{[0-9]+}}]
// CHECK-SAME: ][
// CHECK-SAME:   [0 0 0 0 0 0 0 0 0 0]
// CHECK-SAME: ][
// CHECK-SAME:   [0 0 0 0 0 0 0 0 0 0]
// CHECK-SAME: ]

