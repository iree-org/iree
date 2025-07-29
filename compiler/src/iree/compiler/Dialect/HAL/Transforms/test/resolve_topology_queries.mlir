// RUN: iree-opt --split-input-file --iree-hal-resolve-topology-queries %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-hal-resolve-topology-queries --iree-hal-external-resources-mappable %s | FileCheck %s --check-prefix=EXTERNAL-MAPPABLE

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {transparent_access = true})
  ]>
} {
  util.global private @device_a : !hal.device = #hal.device.target<"device_a">
  util.global private @device_b : !hal.device = #hal.device.target<"device_b">

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
  util.global private @device_a : !hal.device = #hal.device.target<"device_a">
  util.global private @device_b : !hal.device = #hal.device.target<"device_b">
  util.global private @device_c : !hal.device = #hal.device.target<"device_c">

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

util.global private @device_a : !hal.device = #hal.device.target<"local">
util.global private @device_b : !hal.device = #hal.device.target<"local">
// CHECK-LABEL: @resolve_with_same_backend
util.func public @resolve_with_same_backend() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<{{.*}}Device{{.*}}> : i32
  // Should not add mapping usage bit because we are on the same device.
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<{{.*}}Transfer{{.*}}|{{.*}}Dispatch{{.*}}> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      lifetime(transient) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

util.global private @device_a : !hal.device = #hal.device.target<"local", {ordinal = 0 : index}>
util.global private @device_b : !hal.device = #hal.device.target<"local", {ordinal = 1 : index}>
// CHECK-LABEL: @resolve_with_different_ordinals
util.func public @resolve_with_different_ordinals() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<{{.*}}Device{{.*}}> : i32
  // Should not add mapping usage bit because we are on the same device.
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<{{.*}}Transfer{{.*}}|{{.*}}Dispatch{{.*}}> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      lifetime(transient) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

util.global private @device_a : !hal.device = #hal.device.target<"device_a">
util.global private @device_b : !hal.device = #hal.device.target<"device_b">

// CHECK-LABEL: @no_resolve_with_different_backends
util.func public @no_resolve_with_different_backends() -> (i32, i32) {
  // CHECK: %[[MEMORY_TYPES:.+]], %[[BUFFER_USAGE:.+]] = hal.allocator.resolve_memory_properties
  // CHECK-SAME: for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      lifetime(transient) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

util.global private @device_a : !hal.device = #hal.device.target<"device_a">
util.global private @device_b : !hal.device = #hal.device.target<"device_b">
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

// -----

// CHECK-LABEL: @resolve_constant_lifetime
util.func public @resolve_constant_lifetime() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(constant) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @resolve_transient_lifetime
util.func public @resolve_transient_lifetime() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(transient) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @resolve_variable_lifetime
util.func public @resolve_variable_lifetime() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(variable) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @resolve_external_lifetime
// EXTERNAL-MAPPABLE-LABEL: @resolve_external_lifetime
util.func public @resolve_external_lifetime() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
  // EXTERNAL-MAPPABLE-NOT: hal.allocator.resolve_memory_properties
  // EXTERNAL-MAPPABLE-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<"HostVisible|DeviceVisible|DeviceLocal"> : i32
  // EXTERNAL-MAPPABLE-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|MappingScoped|MappingAccessRandom|Mapping"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(external) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  // EXTERNAL-MAPPABLE: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @resolve_staging_lifetime
util.func public @resolve_staging_lifetime() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<"HostVisible|HostCoherent|HostLocal|DeviceVisible"> : i32
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|MappingScoped|MappingAccessRandom|Mapping"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(staging) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @resolve_no_attributes
util.func public @resolve_no_attributes() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK-DAG: %[[MEMORY_TYPES:.+]] = hal.memory_type<
  // CHECK-DAG: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      lifetime(transient) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}
