// An example LSTM exported from a python reference model with dummy weights.

// RUN: iree-run-mlir --Xcompiler,iree-input-type=mhlo --Xcompiler,iree-hal-target-backends=llvm-cpu %s --input="1x5xf32=[0,1,0,3,4]" --input="1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=mhlo --Xcompiler,iree-hal-target-backends=vmvx %s --input="1x5xf32=[0,1,0,3,4]" --input="1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" | FileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=mhlo --Xcompiler,iree-hal-target-backends=vulkan-spirv %s --input="1x5xf32=[0,1,0,3,4]" --input="1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" | FileCheck %s)

// Exported via the XLA HLO Importer
// The resulting MLIR was modified by hand by changing all large constants to be
// splats of 0.42, removing the leading "module" wrapper, removing "name"
// attributes, removing extraneous 0s from float constants, and cleaning up
// extra whitespace. On top of that, the result was further trimmed by removing
// some calls from @main and the call graphs of the removed callees.

func.func private @ForwardLoopCond_gFAnjWGSoLs__.167(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<40xf32>, %arg3: tensor<i64>, %arg4: tensor<74x40xf32>, %arg5: tensor<i64>, %arg6: tensor<1x10xf32>, %arg7: tensor<1x10xf32>, %arg8: tensor<5x1x64xf32>, %arg9: tensor<5x1x1xf32>, %arg10: tensor<5x1x1xf32>, %arg11: tensor<5xi64>, %arg12: tensor<5x1x10xf32>, %arg13: tensor<5x1x10xf32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func private @Forward_o16DF3vQKaI__disable_call_shape_inference_true_.189(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<5x1x64xf32>, %arg3: tensor<5x1x1xf32>, %arg4: tensor<5x1x1xf32>) -> (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>) {
  %cst = arith.constant dense<0x7F800000> : tensor<f32>
  %0 = mhlo.constant dense<0.000000e+00> : tensor<5xf32>
  %cst_0 = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %cst_1 = arith.constant dense<-2147483648> : tensor<i32>
  %cst_2 = arith.constant dense<5> : tensor<i32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<40xf32>
  %cst_3 = arith.constant dense<4.200000e-01> : tensor<74x40xf32>
  %cst_4 = arith.constant dense<0> : tensor<i64>
  %2 = mhlo.constant dense<0> : tensor<5xi64>
  %3 = mhlo.constant dense<0.000000e+00> : tensor<5x1x10xf32>
  %cst_5 = arith.constant dense<1> : tensor<i64>
  %4 = mhlo.constant dense<1.000000e+01> : tensor<1x10xf32>
  %5 = mhlo.constant dense<-1.000000e+01> : tensor<1x10xf32>
  %6 = mhlo.constant dense<1.000000e+00> : tensor<1x10xf32>
  %7 = mhlo.constant dense<0.000000e+00> : tensor<1x10xf32>
  %8 = mhlo.constant dense<5.000000e-01> : tensor<1x10xf32>
  %cst_6 = arith.constant dense<0> : tensor<i32>
  %9 = "mhlo.reduce"(%arg3, %cst) ( {
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):  // no predecessors
    %115 = mhlo.minimum %arg5, %arg6 : tensor<f32>
    "mhlo.return"(%115) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
  %10 = "mhlo.compare"(%9, %0) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %11 = "mhlo.convert"(%10) : (tensor<5xi1>) -> tensor<5xi32>
  %12 = mhlo.multiply %11, %cst_0 : tensor<5xi32>
  %13 = "mhlo.reduce"(%12, %cst_1) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
    %115 = mhlo.maximum %arg5, %arg6 : tensor<i32>
    "mhlo.return"(%115) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %14 = mhlo.subtract %cst_2, %13 : tensor<i32>
  %15 = "mhlo.compare"(%14, %cst_2) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %16 = "mhlo.reverse"(%9) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<5xf32>
  %17 = "mhlo.compare"(%16, %0) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %18 = "mhlo.convert"(%17) : (tensor<5xi1>) -> tensor<5xi32>
  %19 = mhlo.multiply %18, %cst_0 : tensor<5xi32>
  %20 = "mhlo.reduce"(%19, %cst_1) ( {
  ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
    %115 = mhlo.maximum %arg5, %arg6 : tensor<i32>
    "mhlo.return"(%115) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
  %21 = mhlo.subtract %cst_2, %20 : tensor<i32>
  %22 = "mhlo.select"(%15, %cst_6, %21) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %23 = "mhlo.convert"(%22) : (tensor<i32>) -> tensor<i64>
  %24 = mhlo.subtract %cst_2, %14 : tensor<i32>
  %25 = "mhlo.convert"(%24) : (tensor<i32>) -> tensor<i64>
  cf.br ^bb1(%23, %25, %1, %cst_4, %cst_3, %cst_4, %arg0, %arg1, %arg2, %arg3, %arg4, %2, %3, %3 : tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>)
^bb1(%26: tensor<i64>, %27: tensor<i64>, %28: tensor<40xf32>, %29: tensor<i64>, %30: tensor<74x40xf32>, %31: tensor<i64>, %32: tensor<1x10xf32>, %33: tensor<1x10xf32>, %34: tensor<5x1x64xf32>, %35: tensor<5x1x1xf32>, %36: tensor<5x1x1xf32>, %37: tensor<5xi64>, %38: tensor<5x1x10xf32>, %39: tensor<5x1x10xf32>):  // 2 preds: ^bb0, ^bb2
  %40 = call @ForwardLoopCond_gFAnjWGSoLs__.167(%26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tensor<i1>
  %41 = tensor.extract %40[] : tensor<i1>
  cf.cond_br %41, ^bb2(%26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39 : tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>), ^bb3(%26, %31, %32, %33, %37, %38, %39 : tensor<i64>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>)
^bb2(%42: tensor<i64>, %43: tensor<i64>, %44: tensor<40xf32>, %45: tensor<i64>, %46: tensor<74x40xf32>, %47: tensor<i64>, %48: tensor<1x10xf32>, %49: tensor<1x10xf32>, %50: tensor<5x1x64xf32>, %51: tensor<5x1x1xf32>, %52: tensor<5x1x1xf32>, %53: tensor<5xi64>, %54: tensor<5x1x10xf32>, %55: tensor<5x1x10xf32>):  // pred: ^bb1
  %56 = mhlo.add %42, %cst_5 : tensor<i64>
  %57 = "mhlo.gather"(%51, %42) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
    slice_sizes = dense<1> : tensor<3xi64>
  } : (tensor<5x1x1xf32>, tensor<i64>) -> tensor<1x1xf32>
  %58 = "mhlo.reshape"(%57) : (tensor<1x1xf32>) -> tensor<1xf32>
  %59 = "mhlo.broadcast_in_dim"(%58) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %60 = mhlo.multiply %59, %6 : tensor<1x10xf32>
  %61 = "mhlo.compare"(%60, %7) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %62 = "mhlo.gather"(%50, %42) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
    slice_sizes = dense<[1, 1, 64]> : tensor<3xi64>
  } : (tensor<5x1x64xf32>, tensor<i64>) -> tensor<1x64xf32>
  %63 = "mhlo.concatenate"(%62, %49) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
  %64 = "mhlo.dot"(%63, %46) {precision_config = [#mhlo<precision DEFAULT">, #mhlo<"precision DEFAULT>]} : (tensor<1x74xf32>, tensor<74x40xf32>) -> tensor<1x40xf32>
  %65 = "mhlo.reshape"(%44) : (tensor<40xf32>) -> tensor<1x40xf32>
  %66 = mhlo.add %64, %65 : tensor<1x40xf32>
  %67 = "mhlo.slice"(%66) {limit_indices = dense<[1, 30]> : tensor<2xi64>, start_indices = dense<[0, 20]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %68 = mhlo.multiply %67, %8 : tensor<1x10xf32>
  %69 = "mhlo.tanh"(%68) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %70 = mhlo.multiply %69, %8 : tensor<1x10xf32>
  %71 = mhlo.add %70, %8 : tensor<1x10xf32>
  %72 = mhlo.multiply %71, %48 : tensor<1x10xf32>
  %73 = "mhlo.slice"(%66) {limit_indices = dense<[1, 20]> : tensor<2xi64>, start_indices = dense<[0, 10]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %74 = mhlo.multiply %73, %8 : tensor<1x10xf32>
  %75 = "mhlo.tanh"(%74) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %76 = mhlo.multiply %75, %8 : tensor<1x10xf32>
  %77 = mhlo.add %76, %8 : tensor<1x10xf32>
  %78 = "mhlo.slice"(%66) {limit_indices = dense<[1, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %79 = "mhlo.tanh"(%78) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %80 = mhlo.multiply %77, %79 : tensor<1x10xf32>
  %81 = mhlo.add %72, %80 : tensor<1x10xf32>
  %82 = mhlo.minimum %81, %4 : tensor<1x10xf32>
  %83 = mhlo.maximum %82, %5 : tensor<1x10xf32>
  %84 = "mhlo.select"(%61, %48, %83) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %85 = "mhlo.reshape"(%57) : (tensor<1x1xf32>) -> tensor<1xf32>
  %86 = "mhlo.broadcast_in_dim"(%85) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x10xf32>
  %87 = mhlo.multiply %86, %6 : tensor<1x10xf32>
  %88 = "mhlo.compare"(%87, %7) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %89 = "mhlo.slice"(%66) {limit_indices = dense<[1, 40]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %90 = mhlo.multiply %89, %8 : tensor<1x10xf32>
  %91 = "mhlo.tanh"(%90) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %92 = mhlo.multiply %91, %8 : tensor<1x10xf32>
  %93 = mhlo.add %92, %8 : tensor<1x10xf32>
  %94 = "mhlo.tanh"(%83) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  %95 = mhlo.multiply %93, %94 : tensor<1x10xf32>
  %96 = "mhlo.select"(%88, %49, %95) : (tensor<1x10xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %97 = "mhlo.reshape"(%47) : (tensor<i64>) -> tensor<1xi64>
  %98 = "mhlo.reshape"(%42) : (tensor<i64>) -> tensor<1xi64>
  %99 = "mhlo.convert"(%98) : (tensor<1xi64>) -> tensor<1xi32>
  %100 = "mhlo.reshape"(%99) : (tensor<1xi32>) -> tensor<i32>
  %101 = "mhlo.dynamic_update_slice"(%53, %97, %100) : (tensor<5xi64>, tensor<1xi64>, tensor<i32>) -> tensor<5xi64>
  %102 = "mhlo.reshape"(%84) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %103 = "mhlo.reshape"(%99) : (tensor<1xi32>) -> tensor<i32>
  %104 = "mhlo.dynamic_update_slice"(%54, %102, %103, %cst_6, %cst_6) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %105 = "mhlo.reshape"(%96) : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %106 = "mhlo.reshape"(%99) : (tensor<1xi32>) -> tensor<i32>
  %107 = "mhlo.dynamic_update_slice"(%55, %105, %106, %cst_6, %cst_6) : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  cf.br ^bb1(%56, %43, %44, %45, %46, %47, %84, %96, %50, %51, %52, %101, %104, %107 : tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>)
^bb3(%108: tensor<i64>, %109: tensor<i64>, %110: tensor<1x10xf32>, %111: tensor<1x10xf32>, %112: tensor<5xi64>, %113: tensor<5x1x10xf32>, %114: tensor<5x1x10xf32>):  // pred: ^bb1
  return %108, %112, %113, %114, %109, %110, %111 : tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>
}
func.func @main(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5x2x2xf32>) -> tensor<5x1x10xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<1x10xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<5x1x1xf32>
  %2 = "mhlo.reshape"(%arg1) : (tensor<1x5x2x2xf32>) -> tensor<1x5x4xf32>
  %3 = "mhlo.pad"(%2, %cst) {edge_padding_high = dense<[0, 0, 60]> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>} : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x5x64xf32>
  %4 = "mhlo.transpose"(%3) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<1x5x64xf32>) -> tensor<5x1x64xf32>
  %5 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x5xf32>) -> tensor<5x1xf32>
  %6 = "mhlo.reshape"(%5) : (tensor<5x1xf32>) -> tensor<5x1x1xf32>
  %7:7 = call @Forward_o16DF3vQKaI__disable_call_shape_inference_true_.189(%0, %0, %4, %6, %1) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>) -> (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>)
  return %7#3 : tensor<5x1x10xf32>
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
