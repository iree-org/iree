// RUN: iree-opt --split-input-file --iree-codegen-erase-hal-descriptor-type-from-memref --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @hal_uniform_buffer()
func.func @hal_uniform_buffer() {
  // CHECK: %[[P:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32>
  // CHECK: "dialect.memref_consumer"(%[[P]]) : (memref<?x8xf32>) -> ()
  %0 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>)
  "dialect.memref_consumer"(%0) : (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @hal_storage_buffer()
func.func @hal_storage_buffer() {
  // CHECK: %[[P:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32>
  // CHECK: "dialect.memref_consumer"(%[[P]]) : (memref<?x8xf32>) -> ()
  %1 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<storage_buffer>>)
  "dialect.memref_consumer"(%1) : (memref<?x8xf32, #hal.descriptor_type<storage_buffer>>) -> ()
  return
}

// -----

// CHECK: func.func @default_address_space()
func.func @default_address_space() {
  // CHECK: %[[P:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32>
  // CHECK: "dialect.memref_consumer"(%[[P]]) : (memref<?x8xf32>) -> ()
  %2 = "dialect.memref_producer"() : () -> (memref<?x8xf32>)
  "dialect.memref_consumer"(%2) : (memref<?x8xf32>) -> ()
  return
}

// -----

// CHECK: func.func @shared_memory_address_space()
func.func @shared_memory_address_space() {
  // CHECK: %[[P:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32, 3>
  // CHECK: "dialect.memref_consumer"(%[[P]]) : (memref<?x8xf32, 3>) -> ()
  %3 = "dialect.memref_producer"() : () -> (memref<?x8xf32, 3>)
  "dialect.memref_consumer"(%3) : (memref<?x8xf32, 3>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @multi_block()
func.func @multi_block() {
  // CHECK:   %[[P:.+]] = "dialect.memref_producer"() : () -> memref<?x8xf32>
  // CHECK:   cf.br ^[[B:bb.+]](%[[P]] : memref<?x8xf32>)
  // CHECK: ^[[B]](%[[A:.+]]: memref<?x8xf32>):
  // CHECK:   "dialect.memref_consumer"(%[[A]]) : (memref<?x8xf32>) -> ()
  %0 = "dialect.memref_producer"() : () -> (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>)
  cf.br ^bb2(%0: memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>)
^bb2(%1 : memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>):
  "dialect.memref_consumer"(%1) : (memref<?x8xf32, #hal.descriptor_type<uniform_buffer>>) -> ()
  return
}


