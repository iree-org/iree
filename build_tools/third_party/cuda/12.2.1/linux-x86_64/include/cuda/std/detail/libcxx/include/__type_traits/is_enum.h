//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ENUM_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ENUM_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_floating_point.h"
#include "../__type_traits/is_function.h"
#include "../__type_traits/is_integral.h"
#include "../__type_traits/is_member_pointer.h"
#include "../__type_traits/is_pointer.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/is_union.h"
#include "../__type_traits/is_void.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_ENUM) && !defined(_LIBCUDACXX_USE_IS_ENUM_FALLBACK)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_enum
    : public integral_constant<bool, _LIBCUDACXX_IS_ENUM(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_enum_v = _LIBCUDACXX_IS_ENUM(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_enum
    : public integral_constant<bool, !is_void<_Tp>::value             &&
                                     !is_integral<_Tp>::value         &&
                                     !is_floating_point<_Tp>::value   &&
                                     !is_array<_Tp>::value            &&
                                     !is_pointer<_Tp>::value          &&
                                     !is_reference<_Tp>::value        &&
                                     !is_member_pointer<_Tp>::value   &&
                                     !is_union<_Tp>::value            &&
                                     !is_class<_Tp>::value            &&
                                     !is_function<_Tp>::value         > {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_enum_v
    = is_enum<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_ENUM) && !defined(_LIBCUDACXX_USE_IS_ENUM_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ENUM_H
