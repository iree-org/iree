//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_destructible.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/is_scalar.h"
#include "../__type_traits/remove_all_extents.h"
#include "../__utility/declval.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// is_nothrow_destructible

#if defined(_LIBCUDACXX_IS_NOTHROW_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_DESTRUCTIBLE_FALLBACK)

template <class _Tp>
struct is_nothrow_destructible
   : public integral_constant<bool, _LIBCUDACXX_IS_NOTHROW_DESTRUCTIBLE(_Tp)> {};

#elif !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT)

template <bool, class _Tp> struct __libcpp_is_nothrow_destructible;

template <class _Tp>
struct __libcpp_is_nothrow_destructible<false, _Tp>
    : public false_type
{
};

template <class _Tp>
struct __libcpp_is_nothrow_destructible<true, _Tp>
    : public integral_constant<bool, noexcept(_CUDA_VSTD::declval<_Tp>().~_Tp()) >
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_destructible
    : public __libcpp_is_nothrow_destructible<is_destructible<_Tp>::value, _Tp>
{
};

template <class _Tp, size_t _Ns>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_destructible<_Tp[_Ns]>
    : public is_nothrow_destructible<_Tp>
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_destructible<_Tp&>
    : public true_type
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_destructible<_Tp&&>
    : public true_type
{
};

#else

template <class _Tp> struct __libcpp_nothrow_destructor
    : public integral_constant<bool, is_scalar<_Tp>::value ||
                                     is_reference<_Tp>::value> {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_destructible
    : public __libcpp_nothrow_destructor<__remove_all_extents_t<_Tp>> {};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_destructible<_Tp[]>
    : public false_type {};

#endif // defined(_LIBCUDACXX_IS_NOTHROW_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_DESTRUCTIBLE_FALLBACK)

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_destructible_v
    = is_nothrow_destructible<_Tp>::value;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
