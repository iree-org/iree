// RUN: iree-opt --iree-codegen-emulate-narrow-type='load-store-bitwidth=16' --verify-diagnostics %s
// RUN: iree-opt --iree-codegen-emulate-narrow-type='load-store-bitwidth=-1' --verify-diagnostics %s

// expected-error @+1 {{requires an 8-bit or 32-bit storage unit}}
module {}
