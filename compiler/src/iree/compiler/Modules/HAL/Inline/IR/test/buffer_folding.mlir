// RUN: iree-opt --split-input-file --canonicalize -cse %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-hal-inline-conversion --iree-vm-conversion --cse %s | FileCheck %s --check-prefix=VM

// CHECK-LABEL: @FoldBufferLengthOp
// CHECK-SAME: (%[[LENGTH:.+]]: index)
util.func public @FoldBufferLengthOp(%length: index) -> index {
  %c64 = arith.constant 64 : index
  %buffer, %storage = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%length} in !util.buffer
  // CHECK-NOT: hal_inline.buffer.length
  %queried_length = hal_inline.buffer.length<%buffer : !hal.buffer> : index
  // CHECK: return %[[LENGTH]]
  util.return %queried_length : index
}

// -----

// CHECK-LABEL: @FoldBufferStorageOp
util.func public @FoldBufferStorageOp(%length: index) -> !util.buffer {
  %c64 = arith.constant 64 : index
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate
  %buffer, %storage = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%length} in !util.buffer
  // CHECK-NOT: hal_inline.buffer.storage
  %queried_storage = hal_inline.buffer.storage<%buffer : !hal.buffer> : !util.buffer
  // CHECK: return %[[STORAGE]]
  util.return %queried_storage : !util.buffer
}

// -----

// CHECK-LABEL: @FoldBufferViewCreateSubspan
// CHECK-SAME: (%[[BASE_BUFFER:.+]]: !hal.buffer, %[[SUBSPAN_OFFSET:.+]]: index, %[[SUBSPAN_LENGTH:.+]]: index)
util.func public @FoldBufferViewCreateSubspan(%base_buffer: !hal.buffer, %subspan_offset: index, %subspan_length: index) -> !hal.buffer_view {
  %subspan = hal_inline.buffer.subspan<%base_buffer : !hal.buffer>[%subspan_offset, %subspan_length] : !hal.buffer
  // CHECK-DAG: %[[VIEW_OFFSET:.+]] = arith.constant 512
  %view_offset = arith.constant 512 : index
  // CHECK-DAG: %[[VIEW_LENGTH:.+]] = arith.constant 1024
  %view_length = arith.constant 1024 : index
  // CHECK-DAG: %[[FOLDED_OFFSET:.+]] = arith.addi %[[SUBSPAN_OFFSET]], %[[VIEW_OFFSET]]
  // CHECK: = hal_inline.buffer_view.create
  // CHECK-SAME: buffer(%[[BASE_BUFFER]] : !hal.buffer)[%[[FOLDED_OFFSET]], %[[VIEW_LENGTH]]]
  %dim0 = arith.constant 128 : index
  %type = arith.constant 32 : i32
  %encoding = arith.constant 1 : i32
  %view = hal_inline.buffer_view.create buffer(%subspan : !hal.buffer)[%view_offset, %view_length]
                                 shape([%dim0])
                                 type(%type)
                                 encoding(%encoding) : !hal.buffer_view
  util.return %view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @SkipBufferViewBufferOp
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer)
util.func public @SkipBufferViewBufferOp(%buffer: !hal.buffer) -> !hal.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : index
  %view = hal_inline.buffer_view.create buffer(%buffer : !hal.buffer)[%c0, %c64]
                                        shape([%c10, %c11])
                                        type(%c32)
                                        encoding(%c1) : !hal.buffer_view
  %view_buffer = hal_inline.buffer_view.buffer<%view : !hal.buffer_view> : !hal.buffer
  // CHECK: return %[[BUFFER]]
  util.return %view_buffer : !hal.buffer
}

// -----

// Both resources are written before either is read. Merging the allocations
// would let the second write overwrite the first resource's value.
// VM-LABEL: vm.func private @independent_writes(
// VM-SAME: %[[FIRST:[^: ]+]]: i32, %[[SECOND:[^: ]+]]: i32
// VM: %[[A:[^: ]+]]:2 = vm.call @hal_inline.buffer.allocate(
// VM: %[[B:[^: ]+]]:2 = vm.call @hal_inline.buffer.allocate(
// VM: vm.buffer.store.i32 %[[FIRST]], %[[A]]#1[
// VM: vm.buffer.store.i32 %[[SECOND]], %[[B]]#1[
// VM: %[[X:[^ ]+]] = vm.buffer.load.i32 %[[A]]#1[
// VM: %[[Y:[^ ]+]] = vm.buffer.load.i32 %[[B]]#1[
// VM: vm.return %[[X]], %[[Y]]
util.func public @independent_writes(%first: i32, %second: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %a = stream.resource.alloc uninitialized : !stream.resource<staging>{%c4}
  %b = stream.resource.alloc uninitialized : !stream.resource<staging>{%c4}
  stream.resource.store %first, %a[%c0] : i32 -> !stream.resource<staging>{%c4}
  stream.resource.store %second, %b[%c0] : i32 -> !stream.resource<staging>{%c4}
  %x = stream.resource.load %a[%c0] : !stream.resource<staging>{%c4} -> i32
  %y = stream.resource.load %b[%c0] : !stream.resource<staging>{%c4} -> i32
  util.return %x, %y : i32, i32
}

// -----

// The same applies to separate mutable copies of the same initial contents.
// VM-LABEL: vm.func private @independent_initialized_writes(
// VM-SAME: %[[FIRST:[^: ]+]]: i32, %[[SECOND:[^: ]+]]: i32
// VM: %[[A:[^: ]+]]:2 = vm.call @hal_inline.buffer.allocate.initialized(
// VM: %[[B:[^: ]+]]:2 = vm.call @hal_inline.buffer.allocate.initialized(
// VM: vm.buffer.store.i32 %[[FIRST]], %[[A]]#1[
// VM: vm.buffer.store.i32 %[[SECOND]], %[[B]]#1[
// VM: %[[X:[^ ]+]] = vm.buffer.load.i32 %[[A]]#1[
// VM: %[[Y:[^ ]+]] = vm.buffer.load.i32 %[[B]]#1[
// VM: vm.return %[[X]], %[[Y]]
util.func public @independent_initialized_writes(%first: i32, %second: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %source = util.buffer.constant : !util.buffer = dense<0> : tensor<1xi32>
  %a, %sa = hal_inline.buffer.allocate.initialized source(%source : !util.buffer)[%c0, %c4] alignment(%c64) : !hal.buffer{%c4} in !util.buffer
  %b, %sb = hal_inline.buffer.allocate.initialized source(%source : !util.buffer)[%c0, %c4] alignment(%c64) : !hal.buffer{%c4} in !util.buffer
  util.buffer.store %first, %sa[%c0 for %c4] : i32 -> !util.buffer{%c4}
  util.buffer.store %second, %sb[%c0 for %c4] : i32 -> !util.buffer{%c4}
  %x = util.buffer.load %sa[%c0 for %c4] : !util.buffer{%c4} -> i32
  %y = util.buffer.load %sb[%c0 for %c4] : !util.buffer{%c4} -> i32
  util.return %x, %y : i32, i32
}
