// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-transformation-pipeline{serialize-executables=false},canonicalize)' --iree-vm-target-index-bits=64 --mlir-print-local-scope %s | FileCheck %s

module attributes {
  hal.device.targets = [
    #hal.device.target<"vmvx", [
      #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
    ]>
  ]
} {

stream.executable public @add_dispatch_0 {
  stream.executable.export @add_dispatch_0 workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module  {
    func.func @add_dispatch_0(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding, %arg2_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<128xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<128xf32>>
      %arg2 = stream.binding.subspan %arg2_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
      %0 = tensor.empty() : tensor<128xf32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[128], strides=[1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
      %2 = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[128], strides=[1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %2 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> tensor<128xf32>
      flow.dispatch.tensor.store %3, %arg2, offsets=[0], sizes=[128], strides=[1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
      return
    }
  }
}

}

// CHECK-LABEL: hal.executable public @add_dispatch_0
//  CHECK-NEXT:   hal.executable.variant public @vmvx_bytecode_fb target(<"vmvx", "vmvx-bytecode-fb">) {
//  CHECK-NEXT:     hal.executable.export public @add_dispatch_0 ordinal(0)
//  CHECK-SAME:       layout(#hal.pipeline.layout<push_constants = 0, sets = [
//  CHECK-SAME:         <0, bindings = [
//  CHECK-SAME:           <0, storage_buffer>,
//  CHECK-SAME:           <1, storage_buffer>,
//  CHECK-SAME:           <2, storage_buffer>
//       CHECK:     module attributes {vm.toplevel} {
//  CHECK-NEXT:       vm.module public @module {
//  CHECK-NEXT:         vm.func private @add_dispatch_0(
//  CHECK-SAME:             %[[SCRATCHPAD:.+]]: !vm.buffer, %[[CONSTANTS:.+]]: !vm.buffer,
//  CHECK-SAME:             %[[BINDINGS:.+]]: !vm.list<!vm.buffer>
//   CHECK-DAG:           %[[C0_I32:.+]] = vm.const.i32.zero
//   CHECK-DAG:           %[[C0_I64:.+]] = vm.const.i64.zero
//   CHECK-DAG:           %[[C1_I32:.+]] = vm.const.i32 1
//   CHECK-DAG:           %[[C1_I64:.+]] = vm.const.i64 1
//   CHECK-DAG:           %[[C2_I32:.+]] = vm.const.i32 2
//  CHECK-NEXT:           %[[LHS_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %[[C0_I32]] : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           %[[RHS_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %[[C1_I32]] : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           %[[RET_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %[[C2_I32]] : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//       CHECK:           vm.br ^bb1(%[[C0_I64]] : i64)
//  CHECK-NEXT:         ^bb1(%[[IDX:.+]]: i64):
//  CHECK-NEXT:           %slt = vm.cmp.lt.i64.s %[[IDX]], %{{.+}} : i64
//  CHECK-NEXT:           vm.cond_br %slt, ^bb2, ^bb3
//  CHECK-NEXT:         ^bb2:
//  CHECK-NEXT:           %[[ELEMENT_OFFSET:.+]] = vm.add.i64 %[[IDX]], %{{.+}}
//  CHECK-NEXT:           %[[LHS:.+]] = vm.buffer.load.f32 %[[LHS_BUF]][%[[ELEMENT_OFFSET]]] : !vm.buffer -> f32
//  CHECK-NEXT:           %[[RHS:.+]] = vm.buffer.load.f32 %[[RHS_BUF]][%[[ELEMENT_OFFSET]]] : !vm.buffer -> f32
//  CHECK-NEXT:           %[[RET:.+]] = vm.add.f32 %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:           vm.buffer.store.f32 %[[RET]], %[[RET_BUF]][%[[ELEMENT_OFFSET]]] : f32 -> !vm.buffer
//  CHECK-NEXT:           %[[NEXT_IDX:.+]] = vm.add.i64 %[[IDX]], %[[C1_I64]] : i64
//  CHECK-NEXT:           vm.br ^bb1(%[[NEXT_IDX]] : i64)
//  CHECK-NEXT:         ^bb3:
//  CHECK-NEXT:           vm.return
//  CHECK-NEXT:         }
//  CHECK-NEXT:         vm.export @add_dispatch_0
