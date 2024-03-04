//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_CONJUNCTION_H
#define _LIBCUDACXX___TYPE_TRAITS_CONJUNCTION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class...>
using __expand_to_true = true_type;

template <class... _Pred>
_LIBCUDACXX_INLINE_VISIBILITY __expand_to_true<__enable_if_t<_Pred::value>...> __and_helper(int);

template <class...>
_LIBCUDACXX_INLINE_VISIBILITY false_type __and_helper(...);

// _And always performs lazy evaluation of its arguments.
//
// However, `_And<_Pred...>` itself will evaluate its result immediately (without having to
// be instantiated) since it is an alias, unlike `conjunction<_Pred...>`, which is a struct.
// If you want to defer the evaluation of `_And<_Pred...>` itself, use `_Lazy<_And, _Pred...>`.
template <class... _Pred>
using _And _LIBCUDACXX_NODEBUG_TYPE = decltype(__and_helper<_Pred...>(0));

#if _LIBCUDACXX_STD_VER > 11

template <class...>
struct conjunction : true_type {};

template <class _Arg>
struct conjunction<_Arg> : _Arg {};

template <class _Arg, class... _Args>
struct conjunction<_Arg, _Args...> : conditional_t<!bool(_Arg::value), _Arg, conjunction<_Args...>> {};

template <class... _Args>
_LIBCUDACXX_INLINE_VAR constexpr bool conjunction_v = conjunction<_Args...>::value;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_CONJUNCTION_H
