//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_MEMBER_OBJECT_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_MEMBER_OBJECT_POINTER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_member_function_pointer.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_MEMBER_OBJECT_POINTER) && !defined(_LIBCUDACXX_USE_IS_MEMBER_OBJECT_POINTER_FALLBACK)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_member_object_pointer
    : public integral_constant<bool, _LIBCUDACXX_IS_MEMBER_OBJECT_POINTER(_Tp)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_member_object_pointer_v = _LIBCUDACXX_IS_MEMBER_OBJECT_POINTER(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_member_object_pointer
    : public integral_constant<bool, __libcpp_is_member_pointer<__remove_cv_t<_Tp> >::__is_obj >
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_member_object_pointer_v = is_member_object_pointer<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_MEMBER_OBJECT_POINTER) && !defined(_LIBCUDACXX_USE_IS_MEMBER_OBJECT_POINTER_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_MEMBER_FUNCTION_POINTER_H
