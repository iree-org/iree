// RUN: iree-opt --split-input-file --iree-hal-resolve-topology-queries %s | FileCheck %s

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {transparent_access = true})
  ]>
} {

  // CHECK-LABEL: @resolve_with_transparent_access
  util.func public @resolve_with_transparent_access() -> (i32, i32) {
    // CHECK-NOT: hal.allocator.resolve_memory_properties
    // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<{{.*}}Device{{.*}}> : i32
    // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<{{.*}}Transfer{{.*}}|{{.*}}Dispatch{{.*}}|{{.*}}Mapping{{.*}}> : i32
    %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
        for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
        lifetime(transient) : i32, i32
    // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
    util.return %memory_types, %buffer_usage : i32, i32
  }
}

// -----

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {transparent_access = true}),
    (@device_a -> @device_c = {})
  ]>
} {

  // CHECK-LABEL: @no_resolve_with_insufficient_topology
  util.func public @no_resolve_with_insufficient_topology() -> (i32, i32) {
    // CHECK: %[[MEMORY_TYPES:.+]], %[[BUFFER_USAGE:.+]] = hal.allocator.resolve_memory_properties
    %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
        for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>, #hal.device.affinity<@device_c>]>)
        lifetime(transient) : i32, i32
    // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
    util.return %memory_types, %buffer_usage : i32, i32
  }
}



// -----

module {
  // CHECK-LABEL: @no_resolve_without_topology
  util.func public @no_resolve_without_topology() -> (i32, i32) {
    // CHECK: %[[MEMORY_TYPES:.+]], %[[BUFFER_USAGE:.+]] = hal.allocator.resolve_memory_properties
    // CHECK-SAME: for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
    // CHECK-SAME: lifetime(constant) : i32, i32
    %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
        for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
        lifetime(constant) : i32, i32
    // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
    util.return %memory_types, %buffer_usage : i32, i32
  }
}
