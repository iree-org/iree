// RUN: iree-opt --split-input-file --canonicalize -cse %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @fold_binding_subspans_into_dispatch
util.func public @fold_binding_subspans_into_dispatch(
    // CHECK-SAME: %[[EXECUTABLE:.+]]: !hal.executable,
    %executable: !hal.executable,
    // CHECK-SAME: %[[BUFFER:.+]]: !util.buffer, %[[SUBSPAN_OFFSET:.+]]: index, %[[SUBSPAN_LENGTH:.+]]: index
    %buffer: !util.buffer, %subspan_offset: index, %subspan_length: index) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index

  %buffer_size = util.buffer.size %buffer : !util.buffer
  %subspan = util.buffer.subspan %buffer[%subspan_offset] : !util.buffer{%buffer_size} -> !util.buffer{%subspan_length}

  // CHECK-DAG: %[[BINDING_OFFSET:.+]] = arith.constant 100
  %binding_offset = arith.constant 100 : index
  // CHECK-DAG: %[[BINDING_LENGTH:.+]] = arith.constant 128
  %binding_length = arith.constant 128 : index

  // CHECK-DAG: %[[ABSOLUTE_OFFSET:.+]] = arith.addi %[[SUBSPAN_OFFSET]], %[[BINDING_OFFSET]] : index

  // CHECK: hal_loader.executable.dispatch
  hal_loader.executable.dispatch
    executable(%executable : !hal.executable)[%c16]
    workgroups([%c1, %c1, %c1])
    bindings([
      // CHECK: (%[[BUFFER]] : !util.buffer)[%[[ABSOLUTE_OFFSET]], %[[BINDING_LENGTH]]]
      (%subspan : !util.buffer)[%binding_offset, %binding_length]
    ])
  util.return
}
