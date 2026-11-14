// RUN: iree-opt --verify-diagnostics --pass-pipeline='builtin.module(func.func(iree-convert-unsupported-float-arith{include-f16=true include-math=true}))' %s

func.func @dynamic_half_copysign(%magnitude: tensor<?xf16>, %sign: tensor<?xf16>) -> tensor<?xf16> {
  // expected-error @+1 {{requires statically shaped narrow copysign emulation}}
  %result = math.copysign %magnitude, %sign : tensor<?xf16>
  return %result : tensor<?xf16>
}
