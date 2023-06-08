// RUN: iree-opt --iree-codegen-gpu-check-resource-usage %s --verify-diagnostics -split-input-file | FileCheck %s

module {
  // expected-error @+1 {{uses 274432 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc(%arg0: index) {
    %alloc = memref.alloc() : memref<274432xi8, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check we don't choke on memrefs of index.
// CHECK-LABEL: @shared_mem_alloc_index(
module {
  func.func @shared_mem_alloc_index(%arg0: index) {
    %alloc = memref.alloc() : memref<64xindex, #gpu.address_space<workgroup>>
    return
  }
}
