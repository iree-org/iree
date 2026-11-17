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

// i32 to bf16 through f32 rounds twice: 1077936129 becomes 1077936128 in f32,
// a bf16 tie that rounds to 1073741824 instead of 1082130432. Keep the direct
// conversion.
// EMULATED-LABEL: func.func @sitofp_i32_bfloat(
// EMULATED-NEXT: %[[CAST:.+]] = arith.sitofp %arg0 : i32 to bf16
// EMULATED-NEXT: return %[[CAST]] : bf16
func.func @sitofp_i32_bfloat(%input: i32) -> bf16 {
  %result = arith.sitofp %input : i32 to bf16
  return %result : bf16
}

// EMULATED-LABEL: func.func @uitofp_i32_bfloat_vector(
// EMULATED-NEXT: %[[CAST:.+]] = arith.uitofp %arg0 : vector<2xi32> to vector<2xbf16>
// EMULATED-NEXT: return %[[CAST]] : vector<2xbf16>
func.func @uitofp_i32_bfloat_vector(%input: vector<2xi32>) -> vector<2xbf16> {
  %result = arith.uitofp %input : vector<2xi32> to vector<2xbf16>
  return %result : vector<2xbf16>
}

// A 16-bit integer converts to f32 exactly, so one rounding remains.
// EMULATED-LABEL: func.func @sitofp_i16_bfloat(
// EMULATED: arith.sitofp %arg0 : i16 to f32
// EMULATED-NOT: arith.sitofp
// EMULATED: return %{{.+}} : bf16
func.func @sitofp_i16_bfloat(%input: i16) -> bf16 {
  %result = arith.sitofp %input : i16 to bf16
  return %result : bf16
}

// Every i32 that f32 rounds overflows f16 on both paths.
// EMULATED-LABEL: func.func @sitofp_i32_half(
// EMULATED: arith.sitofp %arg0 : i32 to f32
// EMULATED-NOT: arith.sitofp
// EMULATED: return %{{.+}} : f16
func.func @sitofp_i32_half(%input: i32) -> f16 {
  %result = arith.sitofp %input : i32 to f16
  return %result : f16
}
