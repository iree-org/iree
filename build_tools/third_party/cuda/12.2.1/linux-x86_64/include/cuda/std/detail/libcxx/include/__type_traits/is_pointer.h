//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_POINTER) && !defined(_LIBCUDACXX_USE_IS_POINTER_FALLBACK)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_pointer
    : public integral_constant<bool, _LIBCUDACXX_IS_POINTER(_Tp)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_pointer_v = _LIBCUDACXX_IS_POINTER(_Tp);
#endif

#else

template <class _Tp> struct __libcpp_is_pointer       : public false_type {};
template <class _Tp> struct __libcpp_is_pointer<_Tp*> : public true_type {};

template <class _Tp> struct __libcpp_remove_objc_qualifiers { typedef _Tp type; };
#if defined(_LIBCUDACXX_HAS_OBJC_ARC)
template <class _Tp> struct __libcpp_remove_objc_qualifiers<_Tp __strong> { typedef _Tp type; };
template <class _Tp> struct __libcpp_remove_objc_qualifiers<_Tp __weak> { typedef _Tp type; };
template <class _Tp> struct __libcpp_remove_objc_qualifiers<_Tp __autoreleasing> { typedef _Tp type; };
template <class _Tp> struct __libcpp_remove_objc_qualifiers<_Tp __unsafe_unretained> { typedef _Tp type; };
#endif

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_pointer
    : public __libcpp_is_pointer<typename __libcpp_remove_objc_qualifiers<__remove_cv_t<_Tp> >::type> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_pointer_v = is_pointer<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_POINTER) && !defined(_LIBCUDACXX_USE_IS_POINTER_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_H
