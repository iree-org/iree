// RUN: iree-opt %s --split-input-file --iree-transform-dialect-interpreter | FileCheck %s


#matmat_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @wmma(%a: memref<16x16xf32>, %b: memref<16x16xf32>, %c: memref<16x16xf32>) {
  %c0 = arith.constant 0: index
  %cst = arith.constant 0.0: f32
  %va = vector.transfer_read %a[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vb = vector.transfer_read %b[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vc = vector.transfer_read %c[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>

  // CHECK-NOT: vector.contract
  //     CHECK:  gpu.subgroup_mma_compute
  %vres = vector.contract #matmat_trait %va, %vb, %vc
    : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
  vector.transfer_write %vres, %c[%c0, %c0]: vector<16x16xf32>, memref<16x16xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%module: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { unroll_vectors_gpu_wmma } : (!pdl.operation) -> ()
  transform.iree.vector.vector_to_mma_conversion %func { use_wmma } : (!pdl.operation) -> ()

  // Apply canonicalization post-hoc to trigger DCE and pass the test 
  // (i.e. all vector.contract are dead).
  // TODO: consider having the vector_to_mma_conversion do the DCE automatically.
  transform.iree.apply_patterns %func { canonicalization } : (!pdl.operation) -> ()
}

// -----

#matmat_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @mma_sync(%a: memref<16x16xf32>, %b: memref<16x16xf32>, %c: memref<16x16xf32>) {
  %c0 = arith.constant 0: index
  %cst = arith.constant 0.0: f32
  %va = vector.transfer_read %a[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vb = vector.transfer_read %b[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>
  %vc = vector.transfer_read %c[%c0, %c0], %cst: memref<16x16xf32>, vector<16x16xf32>

  // CHECK-NOT: vector.contract
  //     CHECK: nvgpu.mma.sync{{.*}} tf32Enabled}
  %vres = vector.contract #matmat_trait %va, %vb, %vc
    : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
  vector.transfer_write %vres, %c[%c0, %c0]: vector<16x16xf32>, memref<16x16xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%module: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { unroll_vectors_gpu_mma_sync } : (!pdl.operation) -> ()
  transform.iree.vector.vector_to_mma_conversion %func { use_mma_sync } : (!pdl.operation) -> ()

  // Apply canonicalization post-hoc to trigger DCE and pass the test 
  // (i.e. all vector.contract are dead).
  // TODO: consider having the vector_to_mma_conversion do the DCE automatically.
  transform.iree.apply_patterns %func { canonicalization } : (!pdl.operation) -> ()
}
