// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-check-resource-usage))" %s --verify-diagnostics -split-input-file | FileCheck %s

module {
  // expected-error @+1 {{uses 274432 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc(%arg0: index) {
    %alloc = memref.alloc() : memref<274432xi8, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check that we don't choke on memrefs of index.
// CHECK-LABEL: @shared_mem_alloc_index(
module {
  func.func @shared_mem_alloc_index(%arg0: index) {
    %alloc = memref.alloc() : memref<64xindex, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check that memrefs of index return a valid size.
module {
  // expected-error @+1 {{uses 144984 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc_index_too_big(%arg0: index) {
    %alloc = memref.alloc() : memref<18123xindex, #gpu.address_space<workgroup>>
    return
  }
}
