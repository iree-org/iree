//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_ASSIGNABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_ASSIGNABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_scalar.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_NOTHROW_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_ASSIGNABLE_FALLBACK)

template <class _Tp, class _Arg>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_assignable
    : public integral_constant<bool, _LIBCUDACXX_IS_NOTHROW_ASSIGNABLE(_Tp, _Arg)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_assignable_v = _LIBCUDACXX_IS_NOTHROW_ASSIGNABLE(_Tp, _Arg);
#endif

#elif !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT) && !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT_SFINAE)

template <bool, class _Tp, class _Arg> struct __libcpp_is_nothrow_assignable;

template <class _Tp, class _Arg>
struct __libcpp_is_nothrow_assignable<false, _Tp, _Arg>
    : public false_type
{ };

template <class _Tp, class _Arg>
struct __libcpp_is_nothrow_assignable<true, _Tp, _Arg>
    : public integral_constant<bool, noexcept(_CUDA_VSTD::declval<_Tp>() = _CUDA_VSTD::declval<_Arg>()) >
{ };

template <class _Tp, class _Arg>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_assignable
    : public __libcpp_is_nothrow_assignable<is_assignable<_Tp, _Arg>::value, _Tp, _Arg>
{ };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_assignable_v = is_nothrow_assignable<_Tp, _Arg>::value;
#endif

#else

template <class _Tp, class _Arg>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_assignable
    : public false_type {};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_assignable<_Tp&, _Tp>
#if defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_ASSIGN(_Tp)> {};
#else
    : integral_constant<bool, is_scalar<_Tp>::value> {};
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_assignable<_Tp&, _Tp&>
#if defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_ASSIGN(_Tp)> {};
#else
    : integral_constant<bool, is_scalar<_Tp>::value> {};
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_assignable<_Tp&, const _Tp&>
#if defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_ASSIGN(_Tp)> {};
#else
    : integral_constant<bool, is_scalar<_Tp>::value> {};
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)

#ifndef _LIBCUDACXX_HAS_NO_RVALUE_REFERENCES

template <class _Tp>
struct is_nothrow_assignable<_Tp&, _Tp&&>
#if defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_ASSIGN(_Tp)> {};
#else
    : integral_constant<bool, is_scalar<_Tp>::value> {};
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_ASSIGN) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_ASSIGN_FALLBACK)

#endif // _LIBCUDACXX_HAS_NO_RVALUE_REFERENCES

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_assignable_v = is_nothrow_assignable<_Tp, _Arg>::value;
#endif

#endif // !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_ASSIGNABLE_H
