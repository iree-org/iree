// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/executable.h"

#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(executable, method_name) \
  IREE_HAL_VTABLE_DISPATCH(executable, iree_hal_executable, method_name)

IREE_HAL_API_RETAIN_RELEASE(executable);
