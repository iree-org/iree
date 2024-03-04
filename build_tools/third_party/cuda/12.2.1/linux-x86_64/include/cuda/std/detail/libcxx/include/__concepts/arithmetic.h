//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ARITHMETIC_H
#define _LIBCUDACXX___CONCEPTS_ARITHMETIC_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/is_arithmetic.h"
#include "../__type_traits/is_floating_point.h"
#include "../__type_traits/is_integral.h"
#include "../__type_traits/is_signed_integer.h"
#include "../__type_traits/is_signed.h"
#include "../__type_traits/is_unsigned_integer.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

// [concepts.arithmetic], arithmetic concepts

template<class _Tp>
_LIBCUDACXX_CONCEPT integral = _LIBCUDACXX_TRAIT(is_integral, _Tp);

template<class _Tp>
_LIBCUDACXX_CONCEPT signed_integral = integral<_Tp> && _LIBCUDACXX_TRAIT(is_signed, _Tp);

template<class _Tp>
_LIBCUDACXX_CONCEPT unsigned_integral = integral<_Tp> && !signed_integral<_Tp>;

template<class _Tp>
_LIBCUDACXX_CONCEPT floating_point = _LIBCUDACXX_TRAIT(is_floating_point, _Tp);

// Concept helpers for the internal type traits for the fundamental types.
template <class _Tp>
_LIBCUDACXX_CONCEPT __libcpp_unsigned_integer = __libcpp_is_unsigned_integer<_Tp>::value;
template <class _Tp>
_LIBCUDACXX_CONCEPT __libcpp_signed_integer = __libcpp_is_signed_integer<_Tp>::value;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_ARITHMETIC_H
