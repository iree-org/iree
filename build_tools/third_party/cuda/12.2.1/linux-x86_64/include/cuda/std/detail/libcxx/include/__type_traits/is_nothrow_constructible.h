//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_scalar.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

template <class _Tp, class... _Args>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible
    : public integral_constant<bool, _LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE(_Tp, _Args...)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class ..._Args>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_constructible_v = _LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE(_Tp, _Args...);
#endif

#else

#if !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT)

template <bool, bool, class _Tp, class... _Args> struct __libcpp_is_nothrow_constructible;

template <class _Tp, class... _Args>
struct __libcpp_is_nothrow_constructible</*is constructible*/true, /*is reference*/false, _Tp, _Args...>
    : public integral_constant<bool, noexcept(_Tp(_CUDA_VSTD::declval<_Args>()...))>
{
};

template <class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY void __implicit_conversion_to(_Tp) noexcept { }

template <class _Tp, class _Arg>
struct __libcpp_is_nothrow_constructible</*is constructible*/true, /*is reference*/true, _Tp, _Arg>
    : public integral_constant<bool, noexcept(__implicit_conversion_to<_Tp>(_CUDA_VSTD::declval<_Arg>()))>
{
};

template <class _Tp, bool _IsReference, class... _Args>
struct __libcpp_is_nothrow_constructible</*is constructible*/false, _IsReference, _Tp, _Args...>
    : public false_type
{
};

template <class _Tp, class... _Args>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible
    : __libcpp_is_nothrow_constructible<is_constructible<_Tp, _Args...>::value, is_reference<_Tp>::value, _Tp, _Args...>
{
};

template <class _Tp, size_t _Ns>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible<_Tp[_Ns]>
    : __libcpp_is_nothrow_constructible<is_constructible<_Tp>::value, is_reference<_Tp>::value, _Tp>
{
};

#else

template <class _Tp, class... _Args>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible
    : false_type
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible<_Tp>
#if defined(_LIBCUDACXX_HAS_NOTHROW_CONSTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_CONSTRUCTOR_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_CONSTRUCTOR(_Tp)>
#else
    : integral_constant<bool, is_scalar<_Tp>::value>
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_CONSTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_CONSTRUCTOR_FALLBACK)
{
};

template <class _Tp>
#ifndef _LIBCUDACXX_HAS_NO_RVALUE_REFERENCES
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible<_Tp, _Tp&&>
#else
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible<_Tp, _Tp>
#endif
#if defined(_LIBCUDACXX_HAS_NOTHROW_COPY) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_COPY_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_COPY(_Tp)>
#else
    : integral_constant<bool, is_scalar<_Tp>::value>
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_COPY) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_COPY_FALLBACK)
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible<_Tp, const _Tp&>
#if defined(_LIBCUDACXX_HAS_NOTHROW_COPY) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_COPY_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_COPY(_Tp)>
#else
    : integral_constant<bool, is_scalar<_Tp>::value>
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_COPY) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_COPY_FALLBACK)
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_constructible<_Tp, _Tp&>
#if defined(_LIBCUDACXX_HAS_NOTHROW_COPY) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_COPY_FALLBACK)
    : integral_constant<bool, _LIBCUDACXX_HAS_NOTHROW_COPY(_Tp)>
#else
    : integral_constant<bool, is_scalar<_Tp>::value>
#endif // defined(_LIBCUDACXX_HAS_NOTHROW_COPY) && !defined(_LIBCUDACXX_USE_HAS_NOTHROW_COPY_FALLBACK)
{
};

#endif // !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT)

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class ..._Args>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_constructible_v = is_nothrow_constructible<_Tp, _Args...>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
