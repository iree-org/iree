//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_BASE_OF_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_BASE_OF_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

template <class _Bp, class _Dp>
struct _LIBCUDACXX_TEMPLATE_VIS is_base_of
    : public integral_constant<bool, _LIBCUDACXX_IS_BASE_OF(_Bp, _Dp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Bp, class _Dp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_base_of_v = _LIBCUDACXX_IS_BASE_OF(_Bp, _Dp);
#endif

#else  // defined(_LIBCUDACXX_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

namespace __is_base_of_imp
{
template <class _Tp>
struct _Dst
{
    _Dst(const volatile _Tp &);
};
template <class _Tp>
struct _Src
{
    operator const volatile _Tp &();
    template <class _Up> operator const _Dst<_Up> &();
};
template <size_t> struct __one { typedef char type; };
template <class _Bp, class _Dp> typename __one<sizeof(_Dst<_Bp>(declval<_Src<_Dp> >()))>::type __test(int);
template <class _Bp, class _Dp> __two __test(...);
}

template <class _Bp, class _Dp>
struct _LIBCUDACXX_TEMPLATE_VIS is_base_of
    : public integral_constant<bool, is_class<_Bp>::value &&
                                     sizeof(__is_base_of_imp::__test<_Bp, _Dp>(0)) == 2> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Bp, class _Dp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_base_of_v = is_base_of<_Bp, _Dp>::value;
#endif

#endif  // defined(_LIBCUDACXX_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_BASE_OF_H
