// An example LSTM exported from a python reference model with dummy weights.

// RUN: iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=llvm-cpu %s --input="1x5xf32=[0,1,0,3,4]" --input="1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" | FileCheck %s
// RUN: [[ $IREE_VMVX_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=vmvx %s --input="1x5xf32=[0,1,0,3,4]" --input="1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" | FileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --Xcompiler,iree-input-type=stablehlo --Xcompiler,iree-hal-target-backends=vulkan-spirv %s --input="1x5xf32=[0,1,0,3,4]" --input="1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" | FileCheck %s)

// Exported via the XLA HLO Importer
// The resulting MLIR was modified by hand by changing all large constants to be
// splats of 0.42, removing the leading "module" wrapper, removing "name"
// attributes, removing extraneous 0s from float constants, and cleaning up
// extra whitespace. On top of that, the result was further trimmed by removing
// some calls from @main and the call graphs of the removed callees.

func.func private @ForwardLoopCond_gFAnjWGSoLs__.167(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<40xf32>, %arg3: tensor<i64>, %arg4: tensor<74x40xf32>, %arg5: tensor<i64>, %arg6: tensor<1x10xf32>, %arg7: tensor<1x10xf32>, %arg8: tensor<5x1x64xf32>, %arg9: tensor<5x1x1xf32>, %arg10: tensor<5x1x1xf32>, %arg11: tensor<5xi64>, %arg12: tensor<5x1x10xf32>, %arg13: tensor<5x1x10xf32>) -> tensor<i1> {
  %0 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func private @Forward_o16DF3vQKaI__disable_call_shape_inference_true_.189(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<5x1x64xf32>, %arg3: tensor<5x1x1xf32>, %arg4: tensor<5x1x1xf32>) -> (tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>) {
  %cst = arith.constant dense<0x7F800000> : tensor<f32>
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<5xf32>
  %cst_0 = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %cst_1 = arith.constant dense<-2147483648> : tensor<i32>
  %cst_2 = arith.constant dense<5> : tensor<i32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<40xf32>
  %cst_3 = arith.constant dense<4.200000e-01> : tensor<74x40xf32>
  %cst_4 = arith.constant dense<0> : tensor<i64>
  %2 = stablehlo.constant dense<0> : tensor<5xi64>
  %3 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x10xf32>
  %cst_5 = arith.constant dense<1> : tensor<i64>
  %4 = stablehlo.constant dense<1.000000e+01> : tensor<1x10xf32>
  %5 = stablehlo.constant dense<-1.000000e+01> : tensor<1x10xf32>
  %6 = stablehlo.constant dense<1.000000e+00> : tensor<1x10xf32>
  %7 = stablehlo.constant dense<0.000000e+00> : tensor<1x10xf32>
  %8 = stablehlo.constant dense<5.000000e-01> : tensor<1x10xf32>
  %cst_6 = arith.constant dense<0> : tensor<i32>
  %9 = stablehlo.reduce(%arg3 init: %cst) across dimensions = [1, 2] : (tensor<5x1x1xf32>, tensor<f32>) -> tensor<5xf32>
    reducer(%arg5: tensor<f32>, %arg6: tensor<f32>)  {
    %112 = stablehlo.minimum %arg5, %arg6 : tensor<f32>
    stablehlo.return %112 : tensor<f32>
  }
  %10 = stablehlo.compare  EQ, %9, %0 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %11 = stablehlo.convert %10 : (tensor<5xi1>) -> tensor<5xi32>
  %12 = stablehlo.multiply %11, %cst_0 : tensor<5xi32>
  %13 = stablehlo.reduce(%12 init: %cst_1) across dimensions = [0] : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg5: tensor<i32>, %arg6: tensor<i32>)  {
    %112 = stablehlo.maximum %arg5, %arg6 : tensor<i32>
    stablehlo.return %112 : tensor<i32>
  }
  %14 = stablehlo.subtract %cst_2, %13 : tensor<i32>
  %15 = stablehlo.compare  EQ, %14, %cst_2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %16 = stablehlo.reverse %9, dims = [0] : tensor<5xf32>
  %17 = stablehlo.compare  EQ, %16, %0 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
  %18 = stablehlo.convert %17 : (tensor<5xi1>) -> tensor<5xi32>
  %19 = stablehlo.multiply %18, %cst_0 : tensor<5xi32>
  %20 = stablehlo.reduce(%19 init: %cst_1) across dimensions = [0] : (tensor<5xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg5: tensor<i32>, %arg6: tensor<i32>)  {
    %112 = stablehlo.maximum %arg5, %arg6 : tensor<i32>
    stablehlo.return %112 : tensor<i32>
  }
  %21 = stablehlo.subtract %cst_2, %20 : tensor<i32>
  %22 = stablehlo.select %15, %cst_6, %21 : tensor<i1>, tensor<i32>
  %23 = stablehlo.convert %22 : (tensor<i32>) -> tensor<i64>
  %24 = stablehlo.subtract %cst_2, %14 : tensor<i32>
  %25 = stablehlo.convert %24 : (tensor<i32>) -> tensor<i64>
  cf.br ^bb1(%23, %25, %1, %cst_4, %cst_3, %cst_4, %arg0, %arg1, %arg2, %arg3, %arg4, %2, %3, %3 : tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>)
^bb1(%26: tensor<i64>, %27: tensor<i64>, %28: tensor<40xf32>, %29: tensor<i64>, %30: tensor<74x40xf32>, %31: tensor<i64>, %32: tensor<1x10xf32>, %33: tensor<1x10xf32>, %34: tensor<5x1x64xf32>, %35: tensor<5x1x1xf32>, %36: tensor<5x1x1xf32>, %37: tensor<5xi64>, %38: tensor<5x1x10xf32>, %39: tensor<5x1x10xf32>):  // 2 preds: ^bb0, ^bb2
  %40 = call @ForwardLoopCond_gFAnjWGSoLs__.167(%26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39) : (tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>) -> tensor<i1>
  %extracted = tensor.extract %40[] : tensor<i1>
  cf.cond_br %extracted, ^bb2(%26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39 : tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>), ^bb3(%26, %31, %32, %33, %37, %38, %39 : tensor<i64>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>)
^bb2(%41: tensor<i64>, %42: tensor<i64>, %43: tensor<40xf32>, %44: tensor<i64>, %45: tensor<74x40xf32>, %46: tensor<i64>, %47: tensor<1x10xf32>, %48: tensor<1x10xf32>, %49: tensor<5x1x64xf32>, %50: tensor<5x1x1xf32>, %51: tensor<5x1x1xf32>, %52: tensor<5xi64>, %53: tensor<5x1x10xf32>, %54: tensor<5x1x10xf32>):  // pred: ^bb1
  %55 = stablehlo.add %41, %cst_5 : tensor<i64>
  %56 = "stablehlo.gather"(%50, %41) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = dense<1> : tensor<3xi64>} : (tensor<5x1x1xf32>, tensor<i64>) -> tensor<1x1xf32>
  %57 = stablehlo.reshape %56 : (tensor<1x1xf32>) -> tensor<1xf32>
  %58 = stablehlo.broadcast_in_dim %57, dims = [0] : (tensor<1xf32>) -> tensor<1x10xf32>
  %59 = stablehlo.compare  GT, %58, %7 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %60 = "stablehlo.gather"(%49, %41) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = dense<[1, 1, 64]> : tensor<3xi64>} : (tensor<5x1x64xf32>, tensor<i64>) -> tensor<1x64xf32>
  %61 = stablehlo.concatenate %60, %48, dim = 1 : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
  %62 = stablehlo.dot %61, %45, precision = [DEFAULT] : (tensor<1x74xf32>, tensor<74x40xf32>) -> tensor<1x40xf32>
  %63 = stablehlo.reshape %43 : (tensor<40xf32>) -> tensor<1x40xf32>
  %64 = stablehlo.add %62, %63 : tensor<1x40xf32>
  %65 = "stablehlo.slice"(%64) {limit_indices = dense<[1, 30]> : tensor<2xi64>, start_indices = dense<[0, 20]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %66 = stablehlo.multiply %65, %8 : tensor<1x10xf32>
  %67 = stablehlo.tanh %66 : tensor<1x10xf32>
  %68 = stablehlo.multiply %67, %8 : tensor<1x10xf32>
  %69 = stablehlo.add %68, %8 : tensor<1x10xf32>
  %70 = stablehlo.multiply %69, %47 : tensor<1x10xf32>
  %71 = "stablehlo.slice"(%64) {limit_indices = dense<[1, 20]> : tensor<2xi64>, start_indices = dense<[0, 10]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %72 = stablehlo.multiply %71, %8 : tensor<1x10xf32>
  %73 = stablehlo.tanh %72 : tensor<1x10xf32>
  %74 = stablehlo.multiply %73, %8 : tensor<1x10xf32>
  %75 = stablehlo.add %74, %8 : tensor<1x10xf32>
  %76 = "stablehlo.slice"(%64) {limit_indices = dense<[1, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %77 = stablehlo.tanh %76 : tensor<1x10xf32>
  %78 = stablehlo.multiply %75, %77 : tensor<1x10xf32>
  %79 = stablehlo.add %70, %78 : tensor<1x10xf32>
  %80 = stablehlo.minimum %79, %4 : tensor<1x10xf32>
  %81 = stablehlo.maximum %80, %5 : tensor<1x10xf32>
  %82 = stablehlo.select %59, %47, %81 : tensor<1x10xi1>, tensor<1x10xf32>
  %83 = stablehlo.reshape %56 : (tensor<1x1xf32>) -> tensor<1xf32>
  %84 = stablehlo.broadcast_in_dim %83, dims = [0] : (tensor<1xf32>) -> tensor<1x10xf32>
  %85 = stablehlo.compare  GT, %84, %7 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xi1>
  %86 = "stablehlo.slice"(%64) {limit_indices = dense<[1, 40]> : tensor<2xi64>, start_indices = dense<[0, 30]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x40xf32>) -> tensor<1x10xf32>
  %87 = stablehlo.multiply %86, %8 : tensor<1x10xf32>
  %88 = stablehlo.tanh %87 : tensor<1x10xf32>
  %89 = stablehlo.multiply %88, %8 : tensor<1x10xf32>
  %90 = stablehlo.add %89, %8 : tensor<1x10xf32>
  %91 = stablehlo.tanh %81 : tensor<1x10xf32>
  %92 = stablehlo.multiply %90, %91 : tensor<1x10xf32>
  %93 = stablehlo.select %85, %48, %92 : tensor<1x10xi1>, tensor<1x10xf32>
  %94 = stablehlo.reshape %46 : (tensor<i64>) -> tensor<1xi64>
  %95 = stablehlo.reshape %41 : (tensor<i64>) -> tensor<1xi64>
  %96 = stablehlo.convert %95 : (tensor<1xi64>) -> tensor<1xi32>
  %97 = stablehlo.reshape %96 : (tensor<1xi32>) -> tensor<i32>
  %98 = stablehlo.dynamic_update_slice %52, %94, %97 : (tensor<5xi64>, tensor<1xi64>, tensor<i32>) -> tensor<5xi64>
  %99 = stablehlo.reshape %82 : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %100 = stablehlo.reshape %96 : (tensor<1xi32>) -> tensor<i32>
  %101 = stablehlo.dynamic_update_slice %53, %99, %100, %cst_6, %cst_6 : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  %102 = stablehlo.reshape %93 : (tensor<1x10xf32>) -> tensor<1x1x10xf32>
  %103 = stablehlo.reshape %96 : (tensor<1xi32>) -> tensor<i32>
  %104 = stablehlo.dynamic_update_slice %54, %102, %103, %cst_6, %cst_6 : (tensor<5x1x10xf32>, tensor<1x1x10xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5x1x10xf32>
  cf.br ^bb1(%55, %42, %43, %44, %45, %46, %82, %93, %49, %50, %51, %98, %101, %104 : tensor<i64>, tensor<i64>, tensor<40xf32>, tensor<i64>, tensor<74x40xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>, tensor<5x1x64xf32>, tensor<5x1x1xf32>, tensor<5x1x1xf32>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>)
^bb3(%105: tensor<i64>, %106: tensor<i64>, %107: tensor<1x10xf32>, %108: tensor<1x10xf32>, %109: tensor<5xi64>, %110: tensor<5x1x10xf32>, %111: tensor<5x1x10xf32>):  // pred: ^bb1
  return %105, %109, %110, %111, %106, %107, %108 : tensor<i64>, tensor<5xi64>, tensor<5x1x10xf32>, tensor<5x1x10xf32>, tensor<i64>, tensor<1x10xf32>, tensor<1x10xf32>
}
func.func @main(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5x2x2xf32>) -> tensor<5x1x10xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x10xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<5x1x1xf32>
  %2 = stablehlo.reshape %arg1 : (tensor<1x5x2x2xf32>) -> tensor<1x5x4xf32>
  %3 = stablehlo.pad %2, %cst, low = [0, 0, 0], high = [0, 0, 60], interior = [0, 0, 0] : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x5x64xf32>
  %4 = stablehlo.transpose %3, dims = [1, 0, 2] : (tensor<1x5x64xf32>) -> tensor<5x1x64xf32>
  %5 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1x5xf32>) -> tensor<5x1xf32>
  %6 = stablehlo.reshape %5 : (tensor<5x1xf32>) -> tensor<5x1x1xf32>
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
