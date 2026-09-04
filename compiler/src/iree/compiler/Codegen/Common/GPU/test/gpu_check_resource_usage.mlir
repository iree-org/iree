// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-check-resource-usage))" %s --verify-diagnostics -split-input-file | FileCheck %s

module {
  // expected-error @+1 {{uses 274432 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc() {
    memref.alloc() : memref<274432xi8, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check that we don't choke on memrefs of index.
// CHECK-LABEL: func.func @shared_mem_alloc_index()
module {
  func.func @shared_mem_alloc_index() {
    memref.alloc() : memref<64xindex, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Check that memrefs of index return a valid size.
module {
  // expected-error @+1 {{uses 144984 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc_index_too_big() {
    memref.alloc() : memref<18123xindex, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  // expected-error @+1 {{uses 65600 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc_nested_shaped_element() {
    memref.alloc() : memref<1025xvector<16xf32>, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  // expected-error @+1 {{uses 65537 bytes of shared memory; exceeded the limit of 65536 bytes}}
  func.func @shared_mem_alloc_sub_byte_rounds_up() {
    memref.alloc() : memref<524289xi1, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  func.func @shared_mem_alloc_size_overflow() {
    // expected-error @+1 {{shared memory allocation size overflows 64 bits}}
    memref.alloc() : memref<9007199254740991x1024x14x14xf32, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  func.func @shared_mem_alloc_index_overflow() {
    // expected-error @+1 {{shared memory allocation size overflows 64 bits}}
    memref.alloc() : memref<144115188075855872xindex, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  func.func @shared_mem_alloc_nested_shaped_element_overflow() {
    // expected-error @+1 {{shared memory allocation size overflows 64 bits}}
    memref.alloc() : memref<9007199254740991xvector<1024x14x14xf32>, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  func.func @shared_mem_alloc_alignment_overflow() {
    // expected-error @+1 {{shared memory allocation size overflows 64 bits}}
    memref.alloc() alignment = 64 : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    return
  }
}

// -----

module {
  func.func @shared_mem_alloc_cumulative_overflow() {
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    // expected-error @+1 {{cumulative shared memory allocation size overflows 64 bits}}
    memref.alloc() : memref<1152921504606846975xi8, #gpu.address_space<workgroup>>
    return
  }
}
