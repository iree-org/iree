!elem_type_a = type f32
!elem_type_b = type f32
!elem_type_c = type f32
!row_major_A = type tensor<${M}x${K}x!elem_type_a>
!row_major_B = type tensor<${K}x${N}x!elem_type_b>
!row_major_C = type tensor<${M}x${N}x!elem_type_c>

func @init_and_matmul(%a: !row_major_A, %b: !row_major_B, %c: !row_major_C) -> !row_major_C
// TODO: activate manually for now.
// attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
{
  %v0 = constant 0.0 : !elem_type_c
  %d = linalg.fill(%c, %v0) : !row_major_C, !elem_type_c -> !row_major_C
  %e = linalg.matmul ins(%a, %b : !row_major_A, !row_major_B)
    outs(%d: !row_major_C) -> !row_major_C
  return %e : !row_major_C
}

func @print_perf(%iters: index, %total_time: f64) {
  %c2 = constant 2 : index
  %cM = constant ${M} : index
  %cN = constant ${N} : index
  %cK = constant ${K} : index

  %mn = muli %cM, %cN : index
  %mnk = muli %mn, %cK : index

  // 2*M*N*K.
  %flops_per_iter = muli %c2, %mnk : index
  %flops = muli %iters, %flops_per_iter : index
  %flops_i64 = index_cast %flops : index to i64
  %flops_f = sitofp %flops_i64 : i64 to f64
  %flops_per_s = divf %flops_f, %total_time : f64
  vector.print %flops_per_s : f64

  return
}

func @main() {
  %v0 = constant 0.0 : !elem_type_c
  %v1 = constant 1.0 : !elem_type_a
  %v2 = constant 2.0 : !elem_type_b

  %A = linalg.init_tensor [${M}, ${K}] : !row_major_A
  %B = linalg.init_tensor [${K}, ${N}] : !row_major_B
  %C = linalg.init_tensor [${M}, ${N}] : !row_major_C
  %AA = linalg.fill(%A, %v1) : !row_major_A, !elem_type_a -> !row_major_A
  %BB = linalg.fill(%B, %v2) : !row_major_B, !elem_type_b -> !row_major_B
  %CC = linalg.fill(%C, %v0) : !row_major_C, !elem_type_c -> !row_major_C

  %c0 = constant 0: index
  %c1 = constant 1: index
  %iters = constant ${ITERS}: index

  /// Run and dump performance for matmul.
  %t_start_matmul = call @rtclock() : () -> f64
  %res = scf.for %arg0 = %c0 to %iters step %c1 iter_args(%dummy = %CC) -> (!row_major_C) {
    %r = call @init_and_matmul(%AA, %BB, %CC) : (!row_major_A, !row_major_B, !row_major_C) -> (!row_major_C)
    scf.yield %r : !row_major_C
  }
  %t_end_matmul = call @rtclock() : () -> f64
  %tmatmul = subf %t_end_matmul, %t_start_matmul: f64
  call @print_perf(%iters, %tmatmul) : (index, f64) -> ()

  %val = vector.transfer_read %res[%c0, %c0], %v0: !row_major_C, vector<1x1x!elem_type_c>
  vector.print %val: vector<1x1x!elem_type_c>

  // %unrankedRes = tensor.cast %res : !row_major_C to tensor<*x!elem_type_c>
  // call @print_memref_f32(%unrankedRes) : (tensor<*x!elem_type_c>) -> ()

  return
}

func private @rtclock() -> f64

// Abuse reliance on conversions by allowing `tensor<*xf32>`.
// func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
