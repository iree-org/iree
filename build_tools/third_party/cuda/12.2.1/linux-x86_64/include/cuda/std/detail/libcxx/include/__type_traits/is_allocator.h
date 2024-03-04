//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_IS_ALLOCATOR_H
#define _LIBCUDACXX___TYPE_IS_ALLOCATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template<typename _Alloc, typename = void, typename = void>
struct __is_allocator : false_type {};

template<typename _Alloc>
struct __is_allocator<_Alloc,
       __void_t<typename _Alloc::value_type>,
       __void_t<decltype(_CUDA_VSTD::declval<_Alloc&>().allocate(size_t(0)))>
     >
   : true_type {};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_IS_ALLOCATOR_H
