// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="i8=4" --output_types=i | IreeFileCheck %s

// CHECK-LABEL: @extract_element
func @extract_element(%arg0: tensor<i8>) -> i8 {
  %cst = constant dense<1> : tensor<i8>
  %0 = addi %cst, %arg0 : tensor<i8>
  %1 = extract_element %0[] : tensor<i8>
  return %1 : i8
}
// CHECK-NEXT: i8=5
