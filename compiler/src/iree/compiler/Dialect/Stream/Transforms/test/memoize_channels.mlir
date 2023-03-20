// RUN: iree-opt --split-input-file --iree-stream-memoize-channels %s | FileCheck %s

// Tests that default channels are found in callables and deduplicated based on
// their affinity (even if that's inherited from parent ops).

//      CHECK: util.global private @_channel_0 : !stream.channel
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[CHANNEL0:.+]] = stream.channel.create on(#hal.affinity.queue<[0]>) : !stream.channel
// CHECK-NEXT:   util.global.store %[[CHANNEL0]], @_channel_0 : !stream.channel
// CHECK-NEXT:   util.initializer.return
// CHECK-NEXT: }

//      CHECK: util.global private @_channel_1 : !stream.channel
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[CHANNEL1:.+]] = stream.channel.create on(#hal.affinity.queue<[1]>) : !stream.channel
// CHECK-NEXT:   util.global.store %[[CHANNEL1]], @_channel_1 : !stream.channel
// CHECK-NEXT:   util.initializer.return
// CHECK-NEXT: }

// CHECK: util.initializer {
util.initializer {
  // CHECK-NEXT: %[[CHANNEL0:.+]] = util.global.load @_channel_0 : !stream.channel
  %channel0 = stream.channel.default on(#hal.affinity.queue<[0]>) : !stream.channel
  // CHECK-NEXT: util.optimization_barrier %[[CHANNEL0]]
  util.optimization_barrier %channel0 : !stream.channel
  util.initializer.return
}

// CHECK: util.initializer {
util.initializer {
  // CHECK-NEXT: %[[CHANNEL1:.+]] = util.global.load @_channel_1 : !stream.channel
  %channel1 = stream.channel.default on(#hal.affinity.queue<[1]>) : !stream.channel
  // CHECK-NEXT: util.optimization_barrier %[[CHANNEL1]]
  util.optimization_barrier %channel1 : !stream.channel
  util.initializer.return
}

// CHECK: func.func @affinity_func
func.func @affinity_func() -> !stream.channel attributes {
  stream.affinity = #hal.affinity.queue<[0]>
} {
  // CHECK-NEXT: %[[CHANNEL0:.+]] = util.global.load @_channel_0 : !stream.channel
  %channel0 = stream.channel.default : !stream.channel
  // CHECK-NEXT: return %[[CHANNEL0]]
  return %channel0 : !stream.channel
}

// CHECK: func.func @mixed_func
func.func @mixed_func() -> (!stream.channel, !stream.channel) {
  // CHECK-NEXT: %[[CHANNEL0:.+]] = util.global.load @_channel_0 : !stream.channel
  %channel0 = stream.channel.default on(#hal.affinity.queue<[0]>) : !stream.channel
  // CHECK-NEXT: %[[CHANNEL1:.+]] = util.global.load @_channel_1 : !stream.channel
  %channel1 = stream.channel.default on(#hal.affinity.queue<[1]>) : !stream.channel
  // CHECK-NEXT: return %[[CHANNEL0]], %[[CHANNEL1]]
  return %channel0, %channel1 : !stream.channel, !stream.channel
}

// -----

// CHECK: func.func @grouping
func.func @grouping() -> (!stream.channel, !stream.channel, !stream.channel) {
  // CHECK-NEXT: %[[CHANNEL_A0:.+]] = util.global.load @_channel_0 : !stream.channel
  %channel_a0 = stream.channel.default on(#hal.affinity.queue<[0]>) group("a") : !stream.channel
  // CHECK-NEXT: %[[CHANNEL_A1:.+]] = util.global.load @_channel_1 : !stream.channel
  %channel_a1 = stream.channel.default on(#hal.affinity.queue<[1]>) group("a") : !stream.channel
  // CHECK-NEXT: %[[CHANNEL_B0:.+]] = util.global.load @_channel_2 : !stream.channel
  %channel_b0 = stream.channel.default on(#hal.affinity.queue<[0]>) group("b") : !stream.channel
  // CHECK-NEXT: return %[[CHANNEL_A0]], %[[CHANNEL_A1]], %[[CHANNEL_B0]]
  return %channel_a0, %channel_a1, %channel_b0 : !stream.channel, !stream.channel, !stream.channel
}
