//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template<class _Func, class... _Args, class = decltype(std::declval<_Func>()(std::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY true_type __is_callable_helper(int);
template<class...>
_LIBCUDACXX_INLINE_VISIBILITY false_type __is_callable_helper(...);

template<class _Func, class... _Args>
struct __is_callable : decltype(__is_callable_helper<_Func, _Args...>(0)) {};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H
