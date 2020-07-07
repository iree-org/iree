// An example LSTM exported from a python reference model with dummy weights.

// RUN: iree-run-mlir %s -iree-hal-target-backends=vmla -input-value="1x5xf32=[0 1 0 3 4]" -input-value="1x5x2x2xf32=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20]" -export-all=false  | IreeFileCheck %s --implicit-check-not="[" --implicit-check-not="]"

// Exported via the XLA HLO Importer
// The resulting MLIR was modified by hand by changing all large constants to be
// splats of 0.42, removing the leading "module" wrapper, removing "name"
// attributes, removing extraneous 0s from float constants, and cleaning up
// extra whitespace. On top of that, the result was further trimmed by removing
// some calls from @main and the call graphs of the removed callees.

func @Min_reduction.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes { sym_visibility = "private" } {
  %0 = mhlo.minimum %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}
func @Max_reduction.51(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes { sym_visibility = "private" } {
  %0 = mhlo.maximum %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @Max_1_reduction.55(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes { sym_visibility = "private" } {
  %0 = mhlo.maximum %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
func @ForwardLoopCond_gFAnjWGSoLs__.167(%arg0: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>> attributes { sym_visibility = "private" } {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %1 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %3 = "mhlo.tuple"(%2) : (tensor<i1>) -> tuple<tensor<i1>>
  return %3 : tuple<tensor<i1>>
}
func @Forward_o16DF3vQKaI__disable_call_shape_inference_true_.189(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<5x1x64xf32>, %arg3: tensor<5x1x1xf32>, %arg4: tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>> attributes { sym_visibility = "private" } {
  %cst = constant  dense<5> : tensor<i32>
  %0 = "mhlo.convert"(%arg3) : (tensor<5x1x1xf32>) -> tensor<5x1x1xf32>
  %cst_0 = constant dense<0x7F800000> : tensor<f32>
  %1 = "mhlo.convert"(%cst_0) : (tensor<f32>) -> tensor<f32>
  %2 = "mhlo.reduce"(%0, %1) ( {
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
    %42 = mhlo.minimum %arg5, %arg6 : tensor<f32>
    "mhlo.return"(%42) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  %3 = "mhlo.convert"(%2) : (tensor<5xf32>) -> tensor<5xf32>
  %cst_1 = constant  dense<0.000000e+00> : tensor<f32>
  %4 = "mhlo.broadcast_in_dim"(%cst_1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<5xf32>
  %5 = "mhlo.compare"(%3, %4) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %6 = "mhlo.convert"(%5) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_2 = constant  dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %7 = mhlo.multiply %6, %cst_2 : tensor<5xi32>
  %8 = "mhlo.convert"(%7) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_3 = constant dense<-2147483648> : tensor<i32>
  %9 = "mhlo.convert"(%cst_3) : (tensor<i32>) -> tensor<i32>
  %10 = "mhlo.reduce"(%8, %9) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):
    %42 = mhlo.maximum %arg5, %arg6 : tensor<i32>
    "mhlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %11 = "mhlo.convert"(%10) : (tensor<i32>) -> tensor<i32>
  %12 = mhlo.subtract %cst, %11 : tensor<i32>
  %cst_4 = constant dense<5> : tensor<i32>
  %13 = "mhlo.compare"(%12, %cst_4) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %cst_5 = constant dense<0> : tensor<i32>
  %cst_6 = constant dense<5> : tensor<i32>
  %14 = "mhlo.reverse"(%3) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5xf32>
  %cst_7 = constant dense<0.000000e+00> : tensor<f32>
  %15 = "mhlo.broadcast_in_dim"(%cst_7) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<5xf32>
  %16 = "mhlo.compare"(%14, %15) {comparison_direction = "EQ"} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %17 = "mhlo.convert"(%16) : (tensor<5xi1>) -> tensor<5xi32>
  %cst_8 = constant  dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %18 = mhlo.multiply %17, %cst_8 : tensor<5xi32>
  %19 = "mhlo.convert"(%18) : (tensor<5xi32>) -> tensor<5xi32>
  %cst_9 = constant dense<-2147483648> : tensor<i32>
  %20 = "mhlo.convert"(%cst_9) : (tensor<i32>) -> tensor<i32>
  %21 = "mhlo.reduce"(%19, %20) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):
    %42 = mhlo.maximum %arg5, %arg6 : tensor<i32>
    "mhlo.return"(%42) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %22 = "mhlo.convert"(%21) : (tensor<i32>) -> tensor<i32>
  %23 = mhlo.subtract %cst_6, %22 : tensor<i32>
  %24 = "mhlo.select"(%13, %cst_5, %23) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %25 = "mhlo.convert"(%24) : (tensor<i32>) -> tensor<i64>
  %cst_10 = constant dense<5> : tensor<i32>
  %26 = mhlo.subtract %cst_10, %12 : tensor<i32>
  %27 = "mhlo.convert"(%26) : (tensor<i32>) -> tensor<i64>
  %cst_11 = constant dense<0.000000e+00> : tensor<f32>
  %28 = "mhlo.broadcast_in_dim"(%cst_11) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<40xf32>
  %cst_12 = constant  dense<0> : tensor<i64>
  %cst_13 = constant  dense<0.42> : tensor<74x40xf32>
  %cst_14 = constant  dense<0> : tensor<i64>
  %cst_15 = constant  dense<0> : tensor<i64>
  %29 = "mhlo.broadcast_in_dim"(%cst_15) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i64>) -> tensor<5xi64>
  %cst_16 = constant dense<0.000000e+00> : tensor<f32>
  %30 = "mhlo.broadcast_in_dim"(%cst_16) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x1x10xf32>
  %cst_17 = constant dense<0.000000e+00> : tensor<f32>
  %31 = "mhlo.broadcast_in_dim"(%cst_17) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x1x10xf32>
  %32 = "mhlo.tuple"(%25, %27, %28, %cst_12, %cst_13, %cst_14, %arg0, %arg1, %arg2, %arg3, %arg4, %29, %30, %31) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %33 = "mhlo.while"(%32) ( {
  ^bb0(%arg5: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>):
    %42 = call @ForwardLoopCond_gFAnjWGSoLs__.167(%arg5) : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i1>>
    %43 = "mhlo.get_tuple_element"(%42) {index = 0 : i32} : (tuple<tensor<i1>>) -> tensor<i1>
    "mhlo.return"(%43) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg5: tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>):
    %42 = "mhlo.get_tuple_element"(%arg5) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
    %cst_18 = constant dense<1> : tensor<i64>
    %43 = mhlo.add %42, %cst_18 : tensor<i64>
    %44 = "mhlo.get_tuple_element"(%arg5) {index = 1 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
    %45 = "mhlo.get_tuple_element"(%arg5) {index = 2 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<40xf32>
    %46 = "mhlo.get_tuple_element"(%arg5) {index = 3 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
    %47 = "mhlo.get_tuple_element"(%arg5) {index = 4 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<74x40xf32>
    %48 = "mhlo.get_tuple_element"(%arg5) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
    %49 = "mhlo.get_tuple_element"(%arg5) {index = 9 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
    %50 = "mhlo.gather"(%49, %42) {dimension_numbers = {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, start_index_map = dense<0> : tensor<1xi64>}, slice_sizes = dense<1> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>} : (tensor<5x1x1xf32>, tensor<i64>) -> tensor<1x1xf32>
    %51 = "mhlo.reshape"(%50) : (tensor<1x1xf32>) -> tensor<1xf32>
    %52 = "mhlo.broadcast_in_dim"(%51) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
    %cst_19 = constant dense<1.000000e+00> : tensor<f32>
    %53 = "mhlo.broadcast_in_dim"(%cst_19) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %54 = mhlo.multiply %52, %53 : tensor<1x10xf32>
    %cst_20 = constant dense<0.000000e+00> : tensor<f32>
    %55 = "mhlo.broadcast_in_dim"(%cst_20) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %56 = "mhlo.compare"(%54, %55) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
    %57 = "mhlo.get_tuple_element"(%arg5) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
    %cst_21 = constant dense<5.000000e-01> : tensor<f32>
    %58 = "mhlo.broadcast_in_dim"(%cst_21) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %59 = "mhlo.broadcast_in_dim"(%cst_21) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %60 = "mhlo.broadcast_in_dim"(%cst_21) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %61 = "mhlo.get_tuple_element"(%arg5) {index = 8 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x64xf32>
    %62 = "mhlo.gather"(%61, %42) {dimension_numbers = {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 0 : i64, offset_dims = dense<[0, 1]> : tensor<2xi64>, start_index_map = dense<0> : tensor<1xi64>}, slice_sizes = dense<[1, 1, 64]> : tensor<3xi64>} : (tensor<5x1x64xf32>, tensor<i64>) -> tensor<1x64xf32>
    %63 = "mhlo.get_tuple_element"(%arg5) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
    %64 = "mhlo.concatenate"(%62, %63) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
    %65 = "mhlo.dot"(%64, %47) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x74xf32>, tensor<74x40xf32>) -> tensor<1x40xf32>
    %66 = "mhlo.transpose"(%65) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x40xf32>
    %67 = "mhlo.reshape"(%45) : (tensor<40xf32>) -> tensor<1x40xf32>
    %68 = mhlo.add %66, %67 : tensor<1x40xf32>
    %69 = "mhlo.slice"(%68) {limit_indices = dense<[1, 30]> : tensor<2xi64>, start_indices = dense<[0, 20]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
    %70 = mhlo.multiply %60, %69 : tensor<1x10xf32>
    %71 = "mhlo.tanh"(%70) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %72 = mhlo.multiply %59, %71 : tensor<1x10xf32>
    %73 = mhlo.add %58, %72 : tensor<1x10xf32>
    %74 = mhlo.multiply %73, %57 : tensor<1x10xf32>
    %cst_22 = constant dense<5.000000e-01> : tensor<f32>
    %75 = "mhlo.broadcast_in_dim"(%cst_22) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %76 = "mhlo.broadcast_in_dim"(%cst_22) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %77 = "mhlo.broadcast_in_dim"(%cst_22) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %78 = "mhlo.slice"(%68) {limit_indices = dense<[1, 20]> : tensor<2xi64>, start_indices = dense<[0, 10]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
    %79 = mhlo.multiply %77, %78 : tensor<1x10xf32>
    %80 = "mhlo.tanh"(%79) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %81 = mhlo.multiply %76, %80 : tensor<1x10xf32>
    %82 = mhlo.add %75, %81 : tensor<1x10xf32>
    %83 = "mhlo.slice"(%68) {limit_indices = dense<[1, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
    %84 = "mhlo.tanh"(%83) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %85 = mhlo.multiply %82, %84 : tensor<1x10xf32>
    %86 = mhlo.add %74, %85 : tensor<1x10xf32>
    %cst_23 = constant dense<1.000000e+01> : tensor<f32>
    %87 = "mhlo.broadcast_in_dim"(%cst_23) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %88 = mhlo.minimum %86, %87 : tensor<1x10xf32>
    %cst_24 = constant dense<-1.000000e+01> : tensor<f32>
    %89 = "mhlo.broadcast_in_dim"(%cst_24) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %90 = mhlo.maximum %88, %89 : tensor<1x10xf32>
    %91 = "mhlo.select"(%56, %57, %90) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %92 = "mhlo.reshape"(%50) : (tensor<1x1xf32>) -> tensor<1xf32>
    %93 = "mhlo.broadcast_in_dim"(%92) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
    %cst_25 = constant dense<1.000000e+00> : tensor<f32>
    %94 = "mhlo.broadcast_in_dim"(%cst_25) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %95 = mhlo.multiply %93, %94 : tensor<1x10xf32>
    %cst_26 = constant dense<0.000000e+00> : tensor<f32>
    %96 = "mhlo.broadcast_in_dim"(%cst_26) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %97 = "mhlo.compare"(%95, %96) {comparison_direction = "GT"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
    %cst_27 = constant dense<5.000000e-01> : tensor<f32>
    %98 = "mhlo.broadcast_in_dim"(%cst_27) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %99 = "mhlo.broadcast_in_dim"(%cst_27) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %100 = "mhlo.broadcast_in_dim"(%cst_27) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
    %101 = "mhlo.slice"(%68) {limit_indices = dense<[1, 40]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
    %102 = mhlo.multiply %100, %101 : tensor<1x10xf32>
    %103 = "mhlo.tanh"(%102) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %104 = mhlo.multiply %99, %103 : tensor<1x10xf32>
    %105 = mhlo.add %98, %104 : tensor<1x10xf32>
    %106 = "mhlo.tanh"(%90) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %107 = mhlo.multiply %105, %106 : tensor<1x10xf32>
    %108 = "mhlo.select"(%97, %63, %107) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %109 = "mhlo.get_tuple_element"(%arg5) {index = 10 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x1xf32>
    %110 = "mhlo.get_tuple_element"(%arg5) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
    %111 = "mhlo.reshape"(%48) : (tensor<i64>) -> tensor<1xi64>
    %112 = "mhlo.slice"(%111) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %113 = "mhlo.reshape"(%42) : (tensor<i64>) -> tensor<1xi64>
    %114 = "mhlo.concatenate"(%113) {dimension = 0 : i64} : (tensor<1xi64>) -> tensor<1xi64>
    %115 = "mhlo.convert"(%114) : (tensor<1xi64>) -> tensor<1xi32>
    %116 = "mhlo.slice"(%115) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
    %117 = "mhlo.reshape"(%116) : (tensor<1xi32>) -> tensor<i32>
    %118 = "mhlo.dynamic-update-slice"(%110, %112, %117) : (tensor<5xi64>, tensor<1xi64>, tensor<i32>) -> tensor<5xi64>
    %119 = "mhlo.get_tuple_element"(%arg5) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
    %120 = "mhlo.reshape"(%91) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
    %121 = "mhlo.slice"(%120) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
    %122 = "mhlo.slice"(%115) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
    %123 = "mhlo.reshape"(%122) : (tensor<1xi32>) -> tensor<i32>
    %cst_28 = constant dense<0> : tensor<i32>
    %124 = "mhlo.dynamic-update-slice"(%119, %121, %123, %cst_28, %cst_28) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
    %125 = "mhlo.get_tuple_element"(%arg5) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
    %126 = "mhlo.reshape"(%108) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
    %127 = "mhlo.slice"(%126) {limit_indices = dense<[1, 1, 10]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<1x1x10xf32>) -> tensor<1x1x10xf32>
    %128 = "mhlo.slice"(%115) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
    %129 = "mhlo.reshape"(%128) : (tensor<1xi32>) -> tensor<i32>
    %cst_29 = constant dense<0> : tensor<i32>
    %130 = "mhlo.dynamic-update-slice"(%125, %127, %129, %cst_29, %cst_29) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
    %131 = "mhlo.tuple"(%43, %44, %45, %46, %47, %48, %91, %108, %61, %49, %109, %118, %124, %130) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
    "mhlo.return"(%131) : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> ()
  }) : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>
  %34 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %35 = "mhlo.get_tuple_element"(%33) {index = 11 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5xi64>
  %36 = "mhlo.get_tuple_element"(%33) {index = 12 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %37 = "mhlo.get_tuple_element"(%33) {index = 13 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<5x1x10xf32>
  %38 = "mhlo.get_tuple_element"(%33) {index = 5 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<i64>
  %39 = "mhlo.get_tuple_element"(%33) {index = 6 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %40 = "mhlo.get_tuple_element"(%33) {index = 7 : i32} : (tuple<tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>>) -> tensor<1x10xf32>
  %41 = "mhlo.tuple"(%34, %35, %36, %37, %38, %39, %40) : (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  return %41 : tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
}

// CHECK-LABEL: EXEC @main
func @main(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5x2x2xf32>) -> tuple<tensor<5x1x10xf32>> attributes { iree.module.export } {
  %cst = constant dense<0.000000e+00> : tensor<f32>
  %0 = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
  %cst_0 = constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.broadcast_in_dim"(%cst_0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
  %cst_1 = constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.broadcast_in_dim"(%cst_1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
  %cst_2 = constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.broadcast_in_dim"(%cst_2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
  %cst_3 = constant dense<0.000000e+00> : tensor<f32>
  %4 = "mhlo.broadcast_in_dim"(%cst_3) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
  %cst_4 = constant dense<0.000000e+00> : tensor<f32>
  %5 = "mhlo.broadcast_in_dim"(%cst_4) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x10xf32>
  %6 = "mhlo.reshape"(%arg1) : (tensor<1x5x2x2xf32>) -> tensor<1x5x2x2xf32>
  %7 = "mhlo.reshape"(%6) : (tensor<1x5x2x2xf32>) -> tensor<1x5x4xf32>
  %cst_5 = constant dense<0.000000e+00> : tensor<f32>
  %8 = "mhlo.pad"(%7, %cst_5) {edge_padding_high = dense<[0, 0, 60]> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>} : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x5x64xf32>
  %9 = "mhlo.transpose"(%8) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<1x5x64xf32>) -> tensor<5x1x64xf32>
  %10 = "mhlo.reshape"(%arg0) : (tensor<1x5xf32>) -> tensor<1x5xf32>
  %11 = "mhlo.transpose"(%10) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x5xf32>) -> tensor<5x1xf32>
  %12 = "mhlo.reshape"(%11) : (tensor<5x1xf32>) -> tensor<5x1x1xf32>
  %cst_6 = constant dense<0.000000e+00> : tensor<f32>
  %13 = "mhlo.broadcast_in_dim"(%cst_6) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x1x1xf32>
  %14 = call @Forward_o16DF3vQKaI__disable_call_shape_inference_true_.189(%4, %5, %9, %12, %13) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>) -> tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>
  %21 = "mhlo.get_tuple_element"(%14) {index = 3 : i32} : (tuple<tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>>) -> tensor<5x1x10xf32>
  %22 = "mhlo.copy"(%21) : (tensor<5x1x10xf32>) -> tensor<5x1x10xf32>
  %23 = "mhlo.reshape"(%22) : (tensor<5x1x10xf32>) -> tensor<5x1x10xf32>
  %24 = "mhlo.tuple"(%23) : (tensor<5x1x10xf32>) -> tuple<tensor<5x1x10xf32>>
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

