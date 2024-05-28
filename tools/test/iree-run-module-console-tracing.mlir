// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=abs --input="2xf32=-2 3") 2>&1 | FileCheck %s
// RUN: (iree-compile --iree-hal-target-backends=llvm-cpu %s | iree-run-module --device=local-task --module=- --function=abs --input="2xf32=-2 3") 2>&1 | FileCheck %s

// CHECK: main (iree-run-module-main.c
// CHECK: iree_flags_parse_checked (flags.c
// CHECK: iree_tooling_run_module_with_data (run_module.c
func.func @abs(%input : tensor<2xf32>) -> (tensor<2xf32>) {
  %result = math.absf %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
