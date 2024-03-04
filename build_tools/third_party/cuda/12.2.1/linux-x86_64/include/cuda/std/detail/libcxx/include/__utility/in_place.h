//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_IN_PLACE_H
#define _LIBCUDACXX___UTILITY_IN_PLACE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_reference.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

struct _LIBCUDACXX_TYPE_VIS in_place_t {
    explicit in_place_t() = default;
};
_LIBCUDACXX_CPO_ACCESSIBILITY in_place_t in_place{};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS in_place_type_t {
    explicit in_place_type_t() = default;
};
template <class _Tp>
_LIBCUDACXX_CPO_ACCESSIBILITY in_place_type_t<_Tp> in_place_type{};

template <size_t _Idx>
struct _LIBCUDACXX_TEMPLATE_VIS in_place_index_t {
    explicit in_place_index_t() = default;
};
template <size_t _Idx>
_LIBCUDACXX_CPO_ACCESSIBILITY in_place_index_t<_Idx> in_place_index{};

template <class _Tp> struct __is_inplace_type_imp : false_type {};
template <class _Tp> struct __is_inplace_type_imp<in_place_type_t<_Tp>> : true_type {};

template <class _Tp>
using __is_inplace_type = __is_inplace_type_imp<__remove_cvref_t<_Tp>>;

template <class _Tp> struct __is_inplace_index_imp : false_type {};
template <size_t _Idx> struct __is_inplace_index_imp<in_place_index_t<_Idx>> : true_type {};

template <class _Tp>
using __is_inplace_index = __is_inplace_index_imp<__remove_cvref_t<_Tp>>;

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_IN_PLACE_H
