// RUN: iree-opt --iree-codegen-emulate-narrow-type='load-store-bitwidth=32' --verify-diagnostics %s

#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @subword_offset() {
  %offset = arith.constant 2 : index
  // expected-error @+1 {{requires word-aligned binding offsets for 32-bit storage emulation}}
  %buffer = hal.interface.binding.subspan layout(#layout) binding(0) offset(%offset) : memref<4xi16>
  return
}
