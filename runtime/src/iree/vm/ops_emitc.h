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

// Get an array element
#define EMITC_ARRAY_ELEMENT(array, index) (array)[index]

// Get the address of an array element
#define EMITC_ARRAY_ELEMENT_ADDRESS(array, index) &(array)[index]

// Assign a value to an array at a given index
#define EMITC_ARRAY_ELEMENT_ASSIGN(array, index, value) (array)[index] = (value)

#endif  // IREE_VM_OPS_EMITC_H_
