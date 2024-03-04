//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_DISJUNCTION_H
#define _LIBCUDACXX___TYPE_TRAITS_DISJUNCTION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool>
struct _OrImpl;

template <>
struct _OrImpl<true> {
  template <class _Res, class _First, class... _Rest>
  using _Result _LIBCUDACXX_NODEBUG_TYPE =
      typename _OrImpl<!bool(_First::value) && sizeof...(_Rest) != 0>::template _Result<_First, _Rest...>;
};

template <>
struct _OrImpl<false> {
  template <class _Res, class...>
  using _Result = _Res;
};

// _Or always performs lazy evaluation of its arguments.
//
// However, `_Or<_Pred...>` itself will evaluate its result immediately (without having to
// be instantiated) since it is an alias, unlike `disjunction<_Pred...>`, which is a struct.
// If you want to defer the evaluation of `_Or<_Pred...>` itself, use `_Lazy<_Or, _Pred...>`
// or `disjunction<_Pred...>` directly.
template <class... _Args>
using _Or _LIBCUDACXX_NODEBUG_TYPE = typename _OrImpl<sizeof...(_Args) != 0>::template _Result<false_type, _Args...>;

#if _LIBCUDACXX_STD_VER > 11

#ifdef _LIBCUDACXX_COMPILER_MSVC
template <class... _Args>
struct disjunction : false_type {};

template <class _First, class... _Rest>
struct disjunction<_First, _Rest...> : _OrImpl<true>::template _Result<false_type, _First, _Rest...> {};
#else
template <class... _Args>
struct disjunction : _Or<_Args...> {};
#endif // !MSVC

template <class... _Args>
_LIBCUDACXX_INLINE_VAR constexpr bool disjunction_v = _Or<_Args...>::value;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_DISJUNCTION_H
