// RUN: iree-compile --compile-to=input %s | FileCheck %s
// RUN: iree-compile --compile-to=input --iree-input-type=vhlo %s | FileCheck %s
// RUN: iree-compile --compile-to=input --iree-input-type=stablehlo_xla %s | FileCheck %s

// Run in a fresh compiler process: parsing only VHLO must not leave the
// dynamically selected pipeline without its dependent dialects.
// CHECK-LABEL: util.func public @vhlo_negate(
// CHECK-SAME: %[[INPUT:.+]]: tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK: %[[NEG:.+]] = linalg.generic
// CHECK-SAME: ins(%[[INPUT]] : tensor<3x5xf32>)
// CHECK: arith.negf
// CHECK: util.return %[[NEG]] : tensor<3x5xf32>
vhlo.func_v1 @vhlo_negate(%arg0: !vhlo.tensor_v1<3x5x!vhlo.f32_v1>) -> (!vhlo.tensor_v1<3x5x!vhlo.f32_v1>) {
  %0 = "vhlo.negate_v1"(%arg0) : (!vhlo.tensor_v1<3x5x!vhlo.f32_v1>) -> !vhlo.tensor_v1<3x5x!vhlo.f32_v1>
  "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<3x5x!vhlo.f32_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}
