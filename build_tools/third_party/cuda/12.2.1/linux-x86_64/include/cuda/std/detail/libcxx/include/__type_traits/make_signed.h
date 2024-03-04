//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_MAKE_SIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_MAKE_SIGNED_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/apply_cv.h"
#include "../__type_traits/is_enum.h"
#include "../__type_traits/is_integral.h"
#include "../__type_traits/nat.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/type_list.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_MAKE_SIGNED) && !defined(_LIBCUDACXX_USE_MAKE_SIGNED_FALLBACK)

template <class _Tp>
using __make_signed_t = _LIBCUDACXX_MAKE_SIGNED(_Tp);

#else
typedef
    __type_list<signed char,
    __type_list<signed short,
    __type_list<signed int,
    __type_list<signed long,
    __type_list<signed long long,
#  ifndef _LIBCUDACXX_HAS_NO_INT128
    __type_list<__int128_t,
#  endif
    __nat
#  ifndef _LIBCUDACXX_HAS_NO_INT128
    >
#  endif
    > > > > > __signed_types;

template <class _Tp, bool = is_integral<_Tp>::value || is_enum<_Tp>::value>
struct __make_signed_impl {};

template <class _Tp>
struct __make_signed_impl<_Tp, true>
{
    typedef typename __find_first<__signed_types, sizeof(_Tp)>::type type;
};

template <> struct __make_signed_impl<bool,               true> {};
template <> struct __make_signed_impl<  signed short,     true> {typedef short     type;};
template <> struct __make_signed_impl<unsigned short,     true> {typedef short     type;};
template <> struct __make_signed_impl<  signed int,       true> {typedef int       type;};
template <> struct __make_signed_impl<unsigned int,       true> {typedef int       type;};
template <> struct __make_signed_impl<  signed long,      true> {typedef long      type;};
template <> struct __make_signed_impl<unsigned long,      true> {typedef long      type;};
template <> struct __make_signed_impl<  signed long long, true> {typedef long long type;};
template <> struct __make_signed_impl<unsigned long long, true> {typedef long long type;};
#  ifndef _LIBCUDACXX_HAS_NO_INT128
template <> struct __make_signed_impl<__int128_t,         true> {typedef __int128_t type;};
template <> struct __make_signed_impl<__uint128_t,        true> {typedef __int128_t type;};
#  endif

template <class _Tp>
using __make_signed_t = typename __apply_cv<_Tp, typename __make_signed_impl<__remove_cv_t<_Tp> >::type>::type;

#endif // defined(_LIBCUDACXX_MAKE_SIGNED) && !defined(_LIBCUDACXX_USE_MAKE_SIGNED_FALLBACK)

template <class _Tp>
struct make_signed {
  using type _LIBCUDACXX_NODEBUG_TYPE = __make_signed_t<_Tp>;
};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using make_signed_t = __make_signed_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_MAKE_SIGNED_H
