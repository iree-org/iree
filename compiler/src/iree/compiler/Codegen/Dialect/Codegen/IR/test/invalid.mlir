// RUN: iree-opt --split-input-file --verify-diagnostics %s

module {
  func.func @negative_alignment(%src: memref<?x?xi32>) -> memref<?x?xi32> {
    %0 = iree_codegen.assume_alignment %src, -1 : memref<?x?xi32>
    return %0 : memref<?x?xi32>
  }
}

// -----

module {
  func.func @type_mismatch(%src: memref<?x?xi32>) -> memref<?xi32> {
    %0 = iree_codegen.assume_alignment %src, 1 : memref<?xi32>
    return %0 : memref<?xi32>
  }
}

// -----

module {
  func.func @export_config_invalid_type() attributes {
    // expected-error @+1 {{expected workgroup size to have atmost 3 entries}}
    export_config = #iree_codegen.export_config<workgroup_size = [4, 1, 1, 1]>
  } {
    return
  }
}
