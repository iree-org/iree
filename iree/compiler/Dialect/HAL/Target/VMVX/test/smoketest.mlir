// RUN: iree-opt -split-input-file -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false},canonicalize' -mlir-print-local-scope %s | IreeFileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"vmvx", {
      executable_targets = [
        #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
      ]
    }>
  ]
} {

stream.executable public @add_dispatch_0 {
  stream.executable.export @add_dispatch_0
  builtin.module  {
    func @add_dispatch_0(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding, %arg2_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:16xf32>
      %arg2 = stream.binding.subspan %arg2_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:16xf32>
      %0 = linalg.init_tensor [16] : tensor<16xf32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %2 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %3, %arg2, offsets=[0], sizes=[16], strides=[1] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
      return
    }
  }
}

}

// CHECK-LABEL: hal.executable public @add_dispatch_0
//  CHECK-NEXT:   hal.executable.variant public @vmvx_bytecode_fb, target = <"vmvx", "vmvx-bytecode-fb"> {
//  CHECK-NEXT:     hal.executable.entry_point public @add_dispatch_0 ordinal(0)
//  CHECK-SAME:       layout(#hal.executable.layout<push_constants = 0, sets = [
//  CHECK-SAME:         #hal.descriptor_set.layout<0, bindings = [
//  CHECK-SAME:           #hal.descriptor_set.binding<0, storage_buffer>,
//  CHECK-SAME:           #hal.descriptor_set.binding<1, storage_buffer>,
//  CHECK-SAME:           #hal.descriptor_set.binding<2, storage_buffer>
//       CHECK:     module attributes {vm.toplevel} {
//  CHECK-NEXT:       vm.module public @module {
//  CHECK-NEXT:         vm.func private @add_dispatch_0(
//  CHECK-SAME:             %[[SCRATCHPAD:.+]]: !vm.buffer, %[[CONSTANTS:.+]]: !vm.buffer,
//  CHECK-SAME:             %[[BINDINGS:.+]]: !vm.list<!vm.buffer>
//   CHECK-DAG:           %c16 = vm.const.i32 16 : i32
//   CHECK-DAG:           %zero = vm.const.i32.zero : i32
//   CHECK-DAG:           %c1 = vm.const.i32 1 : i32
//   CHECK-DAG:           %c2 = vm.const.i32 2 : i32
//   CHECK-DAG:           %c4 = vm.const.i32 4 : i32
//  CHECK-NEXT:           %[[LHS_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %zero : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           %[[RHS_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %c1 : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           %[[RET_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %c2 : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           vm.br ^bb1(%zero : i32)
//  CHECK-NEXT:         ^bb1(%[[IDX:.+]]: i32):  // 2 preds: ^bb0, ^bb2
//  CHECK-NEXT:           %slt = vm.cmp.lt.i32.s %[[IDX]], %c16 : i32
//  CHECK-NEXT:           vm.cond_br %slt, ^bb2, ^bb3
//  CHECK-NEXT:         ^bb2:  // pred: ^bb1
//  CHECK-NEXT:           %[[BYTE_OFFSET:.+]] = vm.mul.i32 %[[IDX]], %c4 : i32
//  CHECK-NEXT:           %[[LHS:.+]] = vm.buffer.load.f32 %[[LHS_BUF]][%[[BYTE_OFFSET]]] : !vm.buffer -> f32
//  CHECK-NEXT:           %[[RHS:.+]] = vm.buffer.load.f32 %[[RHS_BUF]][%[[BYTE_OFFSET]]] : !vm.buffer -> f32
//  CHECK-NEXT:           %[[RET:.+]] = vm.add.f32 %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:           vm.buffer.store.f32 %[[RET]], %[[RET_BUF]][%[[BYTE_OFFSET]]] : f32 -> !vm.buffer
//  CHECK-NEXT:           %[[NEXT_IDX:.+]] = vm.add.i32 %[[IDX]], %c1 : i32
//  CHECK-NEXT:           vm.br ^bb1(%[[NEXT_IDX]] : i32)
//  CHECK-NEXT:         ^bb3:  // pred: ^bb1
//  CHECK-NEXT:           vm.return
//  CHECK-NEXT:         }
//  CHECK-NEXT:         vm.export @add_dispatch_0
