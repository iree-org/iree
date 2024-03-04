//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_destructible.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/is_scalar.h"
#include "../__type_traits/remove_all_extents.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_TRIVIALLY_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_DESTRUCTIBLE_FALLBACK)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_trivially_destructible
    : public integral_constant<bool, _LIBCUDACXX_IS_TRIVIALLY_DESTRUCTIBLE(_Tp)> {};

#elif defined(_LIBCUDACXX_HAS_TRIVIAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_TRIVIAL_DESTRUCTOR_FALLBACK)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_trivially_destructible
    : public integral_constant<bool, is_destructible<_Tp>::value && _LIBCUDACXX_HAS_TRIVIAL_DESTRUCTOR(_Tp)> {};

#else

template <class _Tp> struct __libcpp_trivial_destructor
    : public integral_constant<bool, is_scalar<_Tp>::value ||
                                     is_reference<_Tp>::value> {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_trivially_destructible
    : public __libcpp_trivial_destructor<__remove_all_extents_t<_Tp>> {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_trivially_destructible<_Tp[]>
    : public false_type {};

#endif // defined(_LIBCUDACXX_HAS_TRIVIAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_TRIVIAL_DESTRUCTOR_FALLBACK)

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_trivially_destructible_v = is_trivially_destructible<_Tp>::value;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H
