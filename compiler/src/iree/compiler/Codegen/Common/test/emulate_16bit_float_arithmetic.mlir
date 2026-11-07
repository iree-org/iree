// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-convert-unsupported-float-arith{include-f16=true include-bf16=true include-math=true}),func.func(iree-convert-unsupported-float-arith{include-f16=false include-bf16=false include-math=false}))' %s | FileCheck %s --check-prefix=EMULATED
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-convert-unsupported-float-arith))' %s | FileCheck %s --check-prefix=NATIVE

// A later pass instance must not reset the earlier instance's option.
// EMULATED-LABEL: func.func @add_half(
// EMULATED-DAG: arith.bitcast %arg0 : f16 to i16
// EMULATED-DAG: arith.bitcast %arg1 : f16 to i16
// EMULATED: %[[SUM:.+]] = arith.addf %{{.+}}, %{{.+}} : f32
// EMULATED: arith.bitcast %[[SUM]] : f32 to i32
// EMULATED: arith.maxsi
// EMULATED: arith.minsi
// EMULATED: arith.trunci %{{.+}} : i32 to i16
// EMULATED: %[[RESULT:.+]] = arith.bitcast %{{.+}} : i16 to f16
// EMULATED: return %[[RESULT]] : f16
// NATIVE-LABEL: func.func @add_half(
// NATIVE-NEXT: %[[SUM:.+]] = arith.addf %arg0, %arg1 : f16
// NATIVE-NEXT: return %[[SUM]] : f16
func.func @add_half(%lhs: f16, %rhs: f16) -> f16 {
  %sum = arith.addf %lhs, %rhs : f16
  return %sum : f16
}


// EMULATED-LABEL: func.func @add_bfloat(
// EMULATED-DAG: arith.bitcast %arg0 : bf16 to i16
// EMULATED-DAG: arith.bitcast %arg1 : bf16 to i16
// EMULATED: %[[SUM:.+]] = arith.addf %{{.+}}, %{{.+}} : f32
// EMULATED: arith.bitcast %[[SUM]] : f32 to i32
// EMULATED: %[[RESULT:.+]] = arith.bitcast %{{.+}} : i16 to bf16
// EMULATED: return %[[RESULT]] : bf16
// NATIVE-LABEL: func.func @add_bfloat(
// NATIVE-NEXT: %[[SUM:.+]] = arith.addf %arg0, %arg1 : bf16
// NATIVE-NEXT: return %[[SUM]] : bf16
func.func @add_bfloat(%lhs: bf16, %rhs: bf16) -> bf16 {
  %sum = arith.addf %lhs, %rhs : bf16
  return %sum : bf16
}

// EMULATED-LABEL: func.func @sqrt_bfloat(
// EMULATED: arith.bitcast %arg0 : bf16 to i16
// EMULATED: %[[ROOT:.+]] = math.sqrt %{{.+}} : f32
// EMULATED: arith.bitcast %[[ROOT]] : f32 to i32
// EMULATED: %[[RESULT:.+]] = arith.bitcast %{{.+}} : i16 to bf16
// EMULATED: return %[[RESULT]] : bf16
// NATIVE-LABEL: func.func @sqrt_bfloat(
// NATIVE-NEXT: %[[ROOT:.+]] = math.sqrt %arg0 : bf16
// NATIVE-NEXT: return %[[ROOT]] : bf16
func.func @sqrt_bfloat(%input: bf16) -> bf16 {
  %root = math.sqrt %input : bf16
  return %root : bf16
}

// NaN sign operands and magnitude payloads must survive emulation.
// EMULATED-LABEL: func.func @copy_sign_half(
// EMULATED-DAG: %[[VALUE_MASK:.+]] = arith.constant 32767 : i16
// EMULATED-DAG: %[[SIGN_MASK:.+]] = arith.constant -32768 : i16
// EMULATED-DAG: %[[MAG:.+]] = arith.bitcast %arg0 : f16 to i16
// EMULATED-DAG: %[[VALUE:.+]] = arith.andi %[[MAG]], %[[VALUE_MASK]] : i16
// EMULATED-DAG: %[[SIGN:.+]] = arith.bitcast %arg1 : f16 to i16
// EMULATED-DAG: %[[SIGNBIT:.+]] = arith.andi %[[SIGN]], %[[SIGN_MASK]] : i16
// EMULATED: %[[BITS:.+]] = arith.ori %[[VALUE]], %[[SIGNBIT]] : i16
// EMULATED: %[[RESULT:.+]] = arith.bitcast %[[BITS]] : i16 to f16
// EMULATED: return %[[RESULT]] : f16
// NATIVE-LABEL: func.func @copy_sign_half(
// NATIVE: math.copysign %arg0, %arg1 : f16
func.func @copy_sign_half(%magnitude: f16, %sign: f16) -> f16 {
  %result = math.copysign %magnitude, %sign : f16
  return %result : f16
}

// EMULATED-LABEL: func.func @copy_sign_bfloat_vector(
// EMULATED: arith.andi {{.+}} : vector<2xi16>
// EMULATED: arith.ori {{.+}} : vector<2xi16>
// NATIVE-LABEL: func.func @copy_sign_bfloat_vector(
// NATIVE: math.copysign %arg0, %arg1 : vector<2xbf16>
func.func @copy_sign_bfloat_vector(%magnitude: vector<2xbf16>, %sign: vector<2xbf16>) -> vector<2xbf16> {
  %result = math.copysign %magnitude, %sign : vector<2xbf16>
  return %result : vector<2xbf16>
}

// EMULATED-LABEL: func.func @copy_sign_fp8(
// EMULATED: arith.andi {{.+}} : i8
// EMULATED: arith.ori {{.+}} : i8
// NATIVE-LABEL: func.func @copy_sign_fp8(
// NATIVE: math.copysign %arg0, %arg1 : f8E5M2
func.func @copy_sign_fp8(%magnitude: f8E5M2, %sign: f8E5M2) -> f8E5M2 {
  %result = math.copysign %magnitude, %sign : f8E5M2
  return %result : f8E5M2
}

// Unsigned-zero formats cannot use the sign-bit shortcut: it could turn a
// zero result into NaN. Preserve their numerical conversion path.
// EMULATED-LABEL: func.func @copy_sign_fp8_unsigned_zero(
// EMULATED: math.copysign {{.+}} : f32
// NATIVE-LABEL: func.func @copy_sign_fp8_unsigned_zero(
// NATIVE: math.copysign %arg0, %arg1 : f8E5M2FNUZ
func.func @copy_sign_fp8_unsigned_zero(%magnitude: f8E5M2FNUZ, %sign: f8E5M2FNUZ) -> f8E5M2FNUZ {
  %result = math.copysign %magnitude, %sign : f8E5M2FNUZ
  return %result : f8E5M2FNUZ
}
