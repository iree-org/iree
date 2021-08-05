// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_OPS_EMITC_H_
#define IREE_VM_OPS_EMITC_H_

// This file contains utility functions and macros used for things that EmitC
// can't handle directly.

// Assign a value through a pointer variable
#define EMITC_DEREF_ASSIGN(ptr, value) *(ptr) = (value)

// Access a member of a struct
#define EMITC_STRUCT_MEMBER(struct, member) (struct).member

// Access a member of a pointer to a struct
#define EMITC_STRUCT_PTR_MEMBER(struct, member) (struct)->member

// Get the address of an array element
#define EMITC_ARRAY_ELEMENT_ADDRESS(array, index) &(array)[index]

// Unary operations
#define EMITC_CAST(arg, type) ((type)(arg))
#define EMITC_NOT(arg) (!(arg))

// Binary operations
#define EMITC_AND(lhs, rhs) ((lhs) && (rhs))
#define EMITC_EQ(lhs, rhs) ((lhs) == (rhs))
#define EMITC_NE(lhs, rhs) ((lhs) != (rhs))
#define EMITC_OR(lhs, rhs) ((lhs) || (rhs))

#endif  // IREE_VM_OPS_EMITC_H_
