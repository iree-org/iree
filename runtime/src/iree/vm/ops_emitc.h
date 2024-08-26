// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_OPS_EMITC_H_
#define IREE_VM_OPS_EMITC_H_

// This file contains utility macros used for things that EmitC  can't handle
// directly.

// Assign a value through a pointer variable
#define EMITC_DEREF_ASSIGN_VALUE(ptr, value) *(ptr) = (value)

// Assign a value pointed to by `ptr` through a pointer variable
#define EMITC_DEREF_ASSIGN_PTR(ptr, value) *(ptr) = *(value)

// Call a function pointer with the given arguments
#define EMITC_CALL_INDIRECT(func, ...) (func)(__VA_ARGS__)

#endif  // IREE_VM_OPS_EMITC_H_
