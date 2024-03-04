//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_INTEGRAL) && !defined(_LIBCUDACXX_USE_IS_INTEGRAL_FALLBACK)

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_integral
    : public integral_constant<bool, _LIBCUDACXX_IS_INTEGRAL(_Tp)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_integral_v = _LIBCUDACXX_IS_INTEGRAL(_Tp);
#endif

#else

template <class _Tp> struct __libcpp_is_integral                     : public false_type {};
template <>          struct __libcpp_is_integral<bool>               : public true_type {};
template <>          struct __libcpp_is_integral<char>               : public true_type {};
template <>          struct __libcpp_is_integral<signed char>        : public true_type {};
template <>          struct __libcpp_is_integral<unsigned char>      : public true_type {};
template <>          struct __libcpp_is_integral<wchar_t>            : public true_type {};
#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
template <>          struct __libcpp_is_integral<char8_t>            : public true_type {};
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
template <>          struct __libcpp_is_integral<char16_t>           : public true_type {};
template <>          struct __libcpp_is_integral<char32_t>           : public true_type {};
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
template <>          struct __libcpp_is_integral<short>              : public true_type {};
template <>          struct __libcpp_is_integral<unsigned short>     : public true_type {};
template <>          struct __libcpp_is_integral<int>                : public true_type {};
template <>          struct __libcpp_is_integral<unsigned int>       : public true_type {};
template <>          struct __libcpp_is_integral<long>               : public true_type {};
template <>          struct __libcpp_is_integral<unsigned long>      : public true_type {};
template <>          struct __libcpp_is_integral<long long>          : public true_type {};
template <>          struct __libcpp_is_integral<unsigned long long> : public true_type {};
#ifndef _LIBCUDACXX_HAS_NO_INT128
template <>          struct __libcpp_is_integral<__int128_t>         : public true_type {};
template <>          struct __libcpp_is_integral<__uint128_t>        : public true_type {};
#endif

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_integral
    : public integral_constant<bool, __libcpp_is_integral<__remove_cv_t<_Tp> >::value>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_integral_v = is_integral<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_INTEGRAL) && !defined(_LIBCUDACXX_USE_IS_INTEGRAL_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H
