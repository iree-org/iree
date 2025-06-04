// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldAllocatorSelect1
// CHECK-SAME: (%[[SAME_DEVICE:.+]]: !hal.device, %[[SAME_AFFINITY:.+]]: i64)
util.func public @FoldAllocatorSelect1(%same_device: !hal.device, %same_affinity: i64) -> (!hal.device, i64) {
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  // CHECK-NOT: hal.allocator.select
  %device, %queue_affinity = hal.allocator.select
      from([
        (%same_device, %same_affinity : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[SAME_DEVICE]], %[[SAME_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectSameDevice
// CHECK-SAME: (%[[SAME_DEVICE:.+]]: !hal.device, %[[AFFINITY_A:.+]]: i64, %[[AFFINITY_B:.+]]: i64)
util.func public @FoldAllocatorSelectSameDevice(%same_device: !hal.device, %affinity_a: i64, %affinity_b: i64) -> (!hal.device, i64) {
  // CHECK: %[[UNUSED:.+]], %[[QUEUE_AFFINITY:.+]] = hal.allocator.select
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  %device, %queue_affinity = hal.allocator.select
      from([
        (%same_device, %affinity_a : !hal.device, i64),
        (%same_device, %affinity_b : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[SAME_DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectSameQueueAffinity
// CHECK-SAME: (%[[DEVICE_A:.+]]: !hal.device, %[[DEVICE_B:.+]]: !hal.device, %[[SAME_AFFINITY:.+]]: i64)
util.func public @FoldAllocatorSelectSameQueueAffinity(%device_a: !hal.device, %device_b: !hal.device, %same_affinity: i64) -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]], %[[UNUSED:.+]] = hal.allocator.select
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  %device, %queue_affinity = hal.allocator.select
      from([
        (%device_a, %same_affinity : !hal.device, i64),
        (%device_b, %same_affinity : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[SAME_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorResolveMemoryPropertiesConstant
util.func public @FoldAllocatorResolveMemoryPropertiesConstant() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|SharingImmutable"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(constant) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @FoldAllocatorResolveMemoryPropertiesTransient
util.func public @FoldAllocatorResolveMemoryPropertiesTransient() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(transient) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @FoldAllocatorResolveMemoryPropertiesVariable
util.func public @FoldAllocatorResolveMemoryPropertiesVariable() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(variable) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @FoldAllocatorResolveMemoryPropertiesExternal
util.func public @FoldAllocatorResolveMemoryPropertiesExternal() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(external) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @FoldAllocatorResolveMemoryPropertiesStaging
util.func public @FoldAllocatorResolveMemoryPropertiesStaging() -> (i32, i32) {
  // CHECK-NOT: hal.allocator.resolve_memory_properties
  // CHECK: %[[MEMORY_TYPES:.+]] = hal.memory_type<"HostVisible|HostCoherent|HostLocal|DeviceVisible"> : i32
  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage|MappingScoped|MappingAccessRandom|Mapping"> : i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.affinity<@device_a>)
      lifetime(staging) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}

// -----

// CHECK-LABEL: @NoFoldAllocatorResolveMemoryPropertiesOptimal
util.func public @NoFoldAllocatorResolveMemoryPropertiesOptimal() -> (i32, i32) {
  // CHECK: %[[MEMORY_TYPES:.+]], %[[BUFFER_USAGE:.+]] = hal.allocator.resolve_memory_properties
  // CHECK-SAME: for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-SAME: lifetime(constant) : i32, i32
  %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties
      for(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      lifetime(constant) : i32, i32
  // CHECK: util.return %[[MEMORY_TYPES]], %[[BUFFER_USAGE]]
  util.return %memory_types, %buffer_usage : i32, i32
}
