// RUN: iree-opt --split-input-file --iree-stream-specialize-encodings %s | FileCheck %s

//------------------------------------------------------------------------------
// Stream ops that have TensorPhaseOp trait. This test suite tests that the
// encoding is updated that carries resolved layouts.
//------------------------------------------------------------------------------

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding_layout = #iree_cpu.vmvx_encoding_layout<>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32]>
module {
  util.global private @device_a = #device_target_local_0_

  util.func public @tensor_sizeof(%d0: index, %d1: index) -> index {
    %size = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
    util.return %size : index
  }
}
// CHECK:       #[[EXECUTABLE:.+]] = #hal.executable.target<"vmvx",
// CHECK:       #[[$ENCODING:.+]] = #iree_encoding.encoding
// CHECK-SAME:    layouts = [#[[EXECUTABLE]]]
// CHECK-LABEL: util.func public @tensor_sizeof
// CHECK:         %[[RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING]]>
// CHECK:         return %[[RES]]
