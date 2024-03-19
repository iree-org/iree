// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @context_resolve
util.func private @context_resolve() {
  // CHECK: = stream.context.resolve : !hal.allocator
  %allocator = stream.context.resolve : !hal.allocator
  // CHECK: = stream.context.resolve on(#hal.affinity.queue<*>) : !hal.device, i64
  %device1, %queue_affinity_any = stream.context.resolve on(#hal.affinity.queue<*>) : !hal.device, i64
  // CHECK: = stream.context.resolve on(#hal.affinity.queue<[4, 5]>) : !hal.device, i64
  %device0, %queue_affinity_45 = stream.context.resolve on(#hal.affinity.queue<[4, 5]>) : !hal.device, i64
  util.return
}
