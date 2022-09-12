// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu | \
// RUN: iree-run-module --entry_function=conv2d_1x230x230x3_7x7x3x64 --device=local-task | \
// RUN: FileCheck %s --check-prefix=EXEC-CONV

func.func @conv2d_1x230x230x3_7x7x3x64() -> tensor<1x112x112x64xf32> {
  %arg0 = util.unfoldable_constant dense<1.0> : tensor<1x230x230x3xf32>
  %arg1 = util.unfoldable_constant dense<0.4> : tensor<7x7x3x64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%1 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  return %2 : tensor<1x112x112x64xf32>
}

//      EXEC-CONV: result[0]: hal.buffer_view
// EXEC-CONV-NEXT: 1x112x112x64xf32=
// EXEC-CONV-SAME: 58.8001 58.8001

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu | \
// RUN: iree-run-module --entry_function=conv2d_1x56x56x64_3x3x64x64_pad --device=local-task | \
// RUN: FileCheck %s --check-prefix=EXEC-CONV-PAD

func.func @conv2d_1x56x56x64_3x3x64x64_pad() -> tensor<1x56x56x64xf32> {
  %arg0 = util.unfoldable_constant dense<1.0> : tensor<1x56x56x64xf32>
  %arg1 = util.unfoldable_constant dense<0.4> : tensor<3x3x64x64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
         ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
                tensor.yield %cst_0 : f32
       } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  %3 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%2, %arg1 : tensor<1x58x58x64xf32>, tensor<3x3x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %3 : tensor<1x56x56x64xf32>
}

//      EXEC-CONV-PAD: result[0]: hal.buffer_view
// EXEC-CONV-PAD-NEXT: 1x56x56x64xf32=
// EXEC-CONV-PAD-SAME: 102.4 102.4
