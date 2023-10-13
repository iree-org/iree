// Regression testcase from https://github.com/openxla/iree/issues/10859

func.func @main(%arg0: tensor<256xi8>, %arg1: tensor<2xi32>, %arg2: tensor<2x32xi8>, %arg3: tensor<32xi32>, %arg4: tensor<32x32xi8>, %arg5: tensor<32xi32>, %arg6: tensor<32x3360xi8>, %arg7: tensor<?x3360xi8>) -> (tensor<?x2xi8>) {
  %0 = tosa.fully_connected %arg7, %arg6, %arg5 {quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>} : (tensor<?x3360xi8>, tensor<32x3360xi8>, tensor<32xi32>) -> tensor<?x32xi32>
  %1 = tosa.rescale %0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32:1101627623>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i32:36>} : (tensor<?x32xi32>) -> tensor<?x32xi8>
  %2 = tosa.clamp %1 {max_fp = 0.000000e+00 : f32, max_int = 127 : i64, min_fp = 0.000000e+00 : f32, min_int = -128 : i64} : (tensor<?x32xi8>) -> tensor<?x32xi8>
  %3 = tosa.fully_connected %2, %arg4, %arg3 {quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>} : (tensor<?x32xi8>, tensor<32x32xi8>, tensor<32xi32>) -> tensor<?x32xi32>
  %4 = tosa.rescale %3 {double_round = true, input_zp = 0 : i32, multiplier = array<i32:1255393165>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i32:35>} : (tensor<?x32xi32>) -> tensor<?x32xi8>
  %5 = tosa.clamp %4 {max_fp = 0.000000e+00 : f32, max_int = 127 : i64, min_fp = 0.000000e+00 : f32, min_int = -128 : i64} : (tensor<?x32xi8>) -> tensor<?x32xi8>
  %6 = tosa.fully_connected %5, %arg2, %arg1 {quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>} : (tensor<?x32xi8>, tensor<2x32xi8>, tensor<2xi32>) -> tensor<?x2xi32>
  %7 = tosa.rescale %6 {double_round = true, input_zp = 0 : i32, multiplier = array<i32:1879992488>, output_zp = 44 : i32, per_channel = false, scale32 = true, shift = array<i32:39>} : (tensor<?x2xi32>) -> tensor<?x2xi8>
  %8 = tosa.table %7, %arg0 : (tensor<?x2xi8>, tensor<256xi8>) -> tensor<?x2xi8>
  return %8 : tensor<?x2xi8>
}
