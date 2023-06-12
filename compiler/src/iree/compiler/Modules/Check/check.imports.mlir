// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

vm.module @check {

vm.import private optional @expect_true(
  %operand : i32
)

vm.import private optional @expect_false(
  %operand : i32
)

vm.import private optional @expect_all_true(
  %device : !vm.ref<!hal.device>,
  %operand : !vm.ref<!hal.buffer_view>,
)

vm.import private optional @expect_eq(
  %device : !vm.ref<!hal.device>,
  %lhs : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>
)

vm.import private optional @expect_almost_eq(
  %device : !vm.ref<!hal.device>,
  %lhs : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>
)

}  // vm.module
