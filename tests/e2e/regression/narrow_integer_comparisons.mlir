// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx --function=i8_equality %s | FileCheck %s --check-prefix=I8-EQUALITY
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --function=i8_equality %s | FileCheck %s --check-prefix=I8-EQUALITY
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx --function=i16_equality %s | FileCheck %s --check-prefix=I16-EQUALITY
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --function=i16_equality %s | FileCheck %s --check-prefix=I16-EQUALITY
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx --function=i8_ordering %s | FileCheck %s --check-prefix=I8-ORDERING
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --function=i8_ordering %s | FileCheck %s --check-prefix=I8-ORDERING
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx --function=i16_ordering %s | FileCheck %s --check-prefix=I16-ORDERING
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --function=i16_ordering %s | FileCheck %s --check-prefix=I16-ORDERING
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx --function=i8_overflow %s | FileCheck %s --check-prefix=I8-OVERFLOW
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --function=i8_overflow %s | FileCheck %s --check-prefix=I8-OVERFLOW
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=vmvx --function=i16_overflow %s | FileCheck %s --check-prefix=I16-OVERFLOW
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-device=local --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu --function=i16_overflow %s | FileCheck %s --check-prefix=I16-OVERFLOW

// Regression for https://github.com/iree-org/iree/issues/25151.
// Unfoldable tensors keep tensor.extract as host loads instead of folding the
// comparisons. Narrow values widened to VM i32 may have different high bits:
// loads are zero-extended, while negative constants are sign-extended.

// I8-EQUALITY-LABEL: EXEC @i8_equality
// I8-EQUALITY-NEXT: result[0]: i32=1
// I8-EQUALITY-NEXT: result[1]: i32=0
// I8-EQUALITY-NEXT: result[2]: i32=0
// I8-EQUALITY-NEXT: result[3]: i32=1
func.func @i8_equality() -> (i1, i1, i1, i1) {
  %equal_tensor = util.unfoldable_constant dense<-4> : tensor<i8>
  %different_tensor = util.unfoldable_constant dense<-3> : tensor<i8>
  %equal = tensor.extract %equal_tensor[] : tensor<i8>
  %different = tensor.extract %different_tensor[] : tensor<i8>
  %cneg4 = arith.constant -4 : i8
  %eq_equal = arith.cmpi eq, %equal, %cneg4 : i8
  %ne_equal = arith.cmpi ne, %equal, %cneg4 : i8
  %eq_different = arith.cmpi eq, %different, %cneg4 : i8
  %ne_different = arith.cmpi ne, %different, %cneg4 : i8
  return %eq_equal, %ne_equal, %eq_different, %ne_different : i1, i1, i1, i1
}

// I16-EQUALITY-LABEL: EXEC @i16_equality
// I16-EQUALITY-NEXT: result[0]: i32=1
// I16-EQUALITY-NEXT: result[1]: i32=0
// I16-EQUALITY-NEXT: result[2]: i32=0
// I16-EQUALITY-NEXT: result[3]: i32=1
func.func @i16_equality() -> (i1, i1, i1, i1) {
  %equal_tensor = util.unfoldable_constant dense<-4> : tensor<i16>
  %different_tensor = util.unfoldable_constant dense<-3> : tensor<i16>
  %equal = tensor.extract %equal_tensor[] : tensor<i16>
  %different = tensor.extract %different_tensor[] : tensor<i16>
  %cneg4 = arith.constant -4 : i16
  %eq_equal = arith.cmpi eq, %equal, %cneg4 : i16
  %ne_equal = arith.cmpi ne, %equal, %cneg4 : i16
  %eq_different = arith.cmpi eq, %different, %cneg4 : i16
  %ne_different = arith.cmpi ne, %different, %cneg4 : i16
  return %eq_equal, %ne_equal, %eq_different, %ne_different : i1, i1, i1, i1
}

// Signed ordering must interpret both loaded values at their original width.
// Unsigned ordering must interpret the negative constant as 252, not 2^32 - 4.
// I8-ORDERING-LABEL: EXEC @i8_ordering
// I8-ORDERING-NEXT: result[0]: i32=1
// I8-ORDERING-NEXT: result[1]: i32=1
// I8-ORDERING-NEXT: result[2]: i32=0
// I8-ORDERING-NEXT: result[3]: i32=0
// I8-ORDERING-NEXT: result[4]: i32=1
func.func @i8_ordering() -> (i1, i1, i1, i1, i1) {
  %negative_tensor = util.unfoldable_constant dense<-3> : tensor<i8>
  %positive_tensor = util.unfoldable_constant dense<3> : tensor<i8>
  %negative = tensor.extract %negative_tensor[] : tensor<i8>
  %positive = tensor.extract %positive_tensor[] : tensor<i8>
  %cneg4 = arith.constant -4 : i8
  %slt = arith.cmpi slt, %negative, %positive : i8
  %sle = arith.cmpi sle, %negative, %positive : i8
  %sgt = arith.cmpi sgt, %negative, %positive : i8
  %sge = arith.cmpi sge, %negative, %positive : i8
  %ugt = arith.cmpi ugt, %negative, %cneg4 : i8
  return %slt, %sle, %sgt, %sge, %ugt : i1, i1, i1, i1, i1
}

// I16-ORDERING-LABEL: EXEC @i16_ordering
// I16-ORDERING-NEXT: result[0]: i32=1
// I16-ORDERING-NEXT: result[1]: i32=1
// I16-ORDERING-NEXT: result[2]: i32=0
// I16-ORDERING-NEXT: result[3]: i32=0
// I16-ORDERING-NEXT: result[4]: i32=1
func.func @i16_ordering() -> (i1, i1, i1, i1, i1) {
  %negative_tensor = util.unfoldable_constant dense<-3> : tensor<i16>
  %positive_tensor = util.unfoldable_constant dense<3> : tensor<i16>
  %negative = tensor.extract %negative_tensor[] : tensor<i16>
  %positive = tensor.extract %positive_tensor[] : tensor<i16>
  %cneg4 = arith.constant -4 : i16
  %slt = arith.cmpi slt, %negative, %positive : i16
  %sle = arith.cmpi sle, %negative, %positive : i16
  %sgt = arith.cmpi sgt, %negative, %positive : i16
  %sge = arith.cmpi sge, %negative, %positive : i16
  %ugt = arith.cmpi ugt, %negative, %cneg4 : i16
  return %slt, %sle, %sgt, %sge, %ugt : i1, i1, i1, i1, i1
}

// Adding loaded -1 and 1 wraps to zero, but a widened VM add can leave bit 8/16
// set. Comparisons must discard those high bits too, not just normalize loads.
// I8-OVERFLOW-LABEL: EXEC @i8_overflow
// I8-OVERFLOW-NEXT: result[0]: i32=1
// I8-OVERFLOW-NEXT: result[1]: i32=0
// I8-OVERFLOW-NEXT: result[2]: i32=0
// I8-OVERFLOW-NEXT: result[3]: i32=1
func.func @i8_overflow() -> (i1, i1, i1, i1) {
  %negative_tensor = util.unfoldable_constant dense<-1> : tensor<i8>
  %positive_tensor = util.unfoldable_constant dense<1> : tensor<i8>
  %negative = tensor.extract %negative_tensor[] : tensor<i8>
  %positive = tensor.extract %positive_tensor[] : tensor<i8>
  %sum = arith.addi %negative, %positive : i8
  %c0 = arith.constant 0 : i8
  %eq = arith.cmpi eq, %sum, %c0 : i8
  %ne = arith.cmpi ne, %sum, %c0 : i8
  %ugt = arith.cmpi ugt, %sum, %positive : i8
  %slt = arith.cmpi slt, %sum, %positive : i8
  return %eq, %ne, %ugt, %slt : i1, i1, i1, i1
}

// I16-OVERFLOW-LABEL: EXEC @i16_overflow
// I16-OVERFLOW-NEXT: result[0]: i32=1
// I16-OVERFLOW-NEXT: result[1]: i32=0
// I16-OVERFLOW-NEXT: result[2]: i32=0
// I16-OVERFLOW-NEXT: result[3]: i32=1
func.func @i16_overflow() -> (i1, i1, i1, i1) {
  %negative_tensor = util.unfoldable_constant dense<-1> : tensor<i16>
  %positive_tensor = util.unfoldable_constant dense<1> : tensor<i16>
  %negative = tensor.extract %negative_tensor[] : tensor<i16>
  %positive = tensor.extract %positive_tensor[] : tensor<i16>
  %sum = arith.addi %negative, %positive : i16
  %c0 = arith.constant 0 : i16
  %eq = arith.cmpi eq, %sum, %c0 : i16
  %ne = arith.cmpi ne, %sum, %c0 : i16
  %ugt = arith.cmpi ugt, %sum, %positive : i16
  %slt = arith.cmpi slt, %sum, %positive : i16
  return %eq, %ne, %ugt, %slt : i1, i1, i1, i1
}
