// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_STATIC_ASSERT_H_
#define IREE_BUILTINS_UKERNEL_STATIC_ASSERT_H_

// IREE_UK_STATIC_ASSERT, a static assert macro usable in C, C++ and
// assembly (though it may evaluate to nothing in assembly).

#if defined(__ASSEMBLER__)
// Note that __STDC__ may also be defined here (when the assembler driver is the
// C compiler driver). So it's easiest to handle __ASSEMBLER__ before __STDC__.
//
// Evaluate to nothing (TODO: If we care for static asserts in assembly, perhaps
// implement them as an assembler .macro?)
#define IREE_UK_STATIC_ASSERT(COND)
#elif defined(__cplusplus)
// Note that depending on the C++ compiler, __STDC__ may or may not be defined
// here! It is defined by G++ but not by Clang++! So, let's also handle
// __cplusplus before __STDC__.
#define IREE_UK_STATIC_ASSERT(COND) static_assert(COND, #COND)
#elif defined(__STDC__) || defined(_MSC_VER)
// Really C, as neither __ASSEMBLER__ nor __cplusplus are defined.
#define IREE_UK_STATIC_ASSERT(COND) _Static_assert(COND, #COND)
#else
#error Expected either __cplusplus or __STDC__ or __ASSEMBLER__ to be defined.
#endif

#endif  // IREE_BUILTINS_UKERNEL_STATIC_ASSERT_H_
