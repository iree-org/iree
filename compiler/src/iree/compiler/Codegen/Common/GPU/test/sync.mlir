#sync = #iree_linalg_ext.requires_sync<#gpu.address_space<workgroup>, 1>

func.func @test(%mem: memref<128xf32>) {
  %shared = memref.alloc() : memref<1xf32, #sync>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %compute_0 = arith.constant dense<0.0> : vector<1xf32>
  %a_0 = vector.transfer_read %mem[%c0], %cst_0 : memref<128xf32>, vector<1xf32>
  vector.transfer_write %a_0, %shared[%c0] : vector<1xf32>, memref<1xf32, #sync>
  %compute_n = scf.for %i = %c1 to %c128 step %c1 iter_args(%compute_i = %compute_0) -> (vector<1xf32>) {
    %a_i = vector.transfer_read %mem[%i], %cst_0 : memref<128xf32>, vector<1xf32>
    %c = vector.transfer_read %shared[%c0], %cst_0 : memref<1xf32, #sync>, vector<1xf32>
    %computei1 = arith.addf %compute_i, %c : vector<1xf32>
    vector.transfer_write %a_i, %shared[%c0] : vector<1xf32>, memref<1xf32, #sync>
    scf.yield %computei1 : vector<1xf32>
  }
  %c = vector.transfer_read %shared[%c0], %cst_0 : memref<1xf32, #sync>, vector<1xf32>
  %compute = arith.addf %compute_n, %c : vector<1xf32>
  vector.transfer_write %compute, %mem[%c0] : vector<1xf32>, memref<128xf32>
  func.return
}
