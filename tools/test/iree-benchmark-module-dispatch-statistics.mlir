// Tests that --dispatch_statistics adds one row per dispatch function directly
// under the benchmark it belongs to (and only if it is passed). The lifecycle
// checks are in runtime/bindings/python/tests/benchmark_lifecycle_test.py.

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=llvm-cpu %s | \
// RUN:  iree-benchmark-module --device=local-task --module=- \
// RUN:                        --function=main \
// RUN:                        --input=64x64xf32=1 --input=64x64xf32=1 \
// RUN:                        --input=64x32xf32=1 \
// RUN:                        --benchmark_min_time=4x) | \
// RUN: FileCheck --check-prefix=DISABLED %s
// DISABLED: BM_main/process_time/real_time
// DISABLED-NOT: main_dispatch_

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=llvm-cpu %s | \
// RUN:  iree-benchmark-module --device=local-task --module=- \
// RUN:                        --function=main \
// RUN:                        --input=64x64xf32=1 --input=64x64xf32=1 \
// RUN:                        --input=64x32xf32=1 \
// RUN:                        --benchmark_min_time=4x \
// RUN:                        --dispatch_statistics) > %t
// RUN: FileCheck --check-prefix=ROWS %s < %t
// RUN: FileCheck --check-prefix=CALLS %s < %t

// The dispatch rows follow the benchmark row in the same table.
// ROWS: BM_main/process_time/real_time {{.+}} 4 items_per_second=
// ROWS-NEXT: BM_main/main_dispatch_{{.+}} 4 calls={{.+}} percent=
// ROWS-NEXT: BM_main/main_dispatch_{{.+}} 4 calls={{.+}} percent=
// ROWS-NOT: BM_

// Calls are counted per benchmark iteration.
// CALLS-DAG: BM_main/main_dispatch_{{[0-9]+}}_matmul_64x64x64_f32 {{.+}} calls=2 percent=
// CALLS-DAG: BM_main/main_dispatch_{{[0-9]+}}_matmul_64x32x64_f32 {{.+}} calls=1 percent=

func.func @main(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>,
                %narrow: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = linalg.matmul ins(%0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  %narrow_empty = tensor.empty() : tensor<64x32xf32>
  %narrow_fill = linalg.fill ins(%zero : f32) outs(%narrow_empty : tensor<64x32xf32>) -> tensor<64x32xf32>
  %2 = linalg.matmul ins(%1, %narrow : tensor<64x64xf32>, tensor<64x32xf32>)
                     outs(%narrow_fill : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %2 : tensor<64x32xf32>
}
