// Tuple flattening must expose bounded tensors before bounds lowering.
// RUN: iree-opt --pass-pipeline='builtin.module(iree-stablehlo-preprocessing-flatten-cfg-tuples,func.func(iree-stablehlo-preprocessing-lower-bounds))' %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.bounds

// CHECK-LABEL: @bounds_tuple
// CHECK: util.assume.int %{{.+}}<umax = 8> : index
// CHECK: return %{{.+}} : tensor<?xf32>
func.func @bounds_tuple(%a: tuple<tensor<?xf32, #stablehlo.bounds<8>>>)
    -> tensor<?xf32, #stablehlo.bounds<8>> {
  %e = "stablehlo.get_tuple_element"(%a) {index = 0 : i32}
    : (tuple<tensor<?xf32, #stablehlo.bounds<8>>>) -> tensor<?xf32, #stablehlo.bounds<8>>
  return %e : tensor<?xf32, #stablehlo.bounds<8>>
}
