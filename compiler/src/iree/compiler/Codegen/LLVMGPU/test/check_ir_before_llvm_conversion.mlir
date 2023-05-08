// RUN: iree-opt --iree-llvmgpu-check-ir-before-llvm-conversion %s --verify-diagnostics -split-input-file

module {
  // expected-error @+1 {{'func.func' op exceeded GPU memory limit of 166912 bytes for function. Got 274432 bytes}}
  func.func @shared_mem_alloc(%arg0: index) {
    %alloc = memref.alloc() : memref<274432xi8, #gpu.address_space<workgroup>>
    return
  }
}
