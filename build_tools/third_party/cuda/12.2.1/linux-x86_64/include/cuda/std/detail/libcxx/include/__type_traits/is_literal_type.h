//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_LITERAL_TYPE
#define _LIBCUDACXX___TYPE_TRAITS_IS_LITERAL_TYPE

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/is_scalar.h"
#include "../__type_traits/remove_all_extents.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_LITERAL) && !defined(_LIBCUDACXX_USE_IS_LITERAL_FALLBACK)
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX17 is_literal_type
    : public integral_constant<bool, _LIBCUDACXX_IS_LITERAL(_Tp)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_INLINE_VAR constexpr bool is_literal_type_v = __is_literal_type(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX17 is_literal_type
    : public integral_constant<bool, is_scalar<__remove_all_extents_t<_Tp>>::value ||
                                     is_reference<__remove_all_extents_t<_Tp>>::value>

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_DEPRECATED_IN_CXX17 _LIBCUDACXX_INLINE_VAR constexpr bool is_literal_type_v
    = is_literal_type<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_LITERAL) && !defined(_LIBCUDACXX_USE_IS_LITERAL_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_LITERAL_TYPE
