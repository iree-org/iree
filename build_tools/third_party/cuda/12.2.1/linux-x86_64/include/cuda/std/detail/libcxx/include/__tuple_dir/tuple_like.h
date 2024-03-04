//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_LIKE_H
#define _LIBCUDACXX___TUPLE_TUPLE_LIKE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__fwd/array.h"
#include "../__fwd/pair.h"
#include "../__fwd/tuple.h"
#include "../__tuple_dir/tuple_types.h"
#include "../__type_traits/integral_constant.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp> struct __tuple_like : false_type {};

template <class _Tp> struct __tuple_like<const _Tp> : public __tuple_like<_Tp> {};
template <class _Tp> struct __tuple_like<volatile _Tp> : public __tuple_like<_Tp> {};
template <class _Tp> struct __tuple_like<const volatile _Tp> : public __tuple_like<_Tp> {};

#ifndef _LIBCUDACXX_CXX03_LANG
template <class... _Tp> struct __tuple_like<tuple<_Tp...> > : true_type {};
#endif

template <class _T1, class _T2> struct __tuple_like<pair<_T1, _T2> > : true_type {};

template <class _Tp, size_t _Size> struct __tuple_like<array<_Tp, _Size> > : true_type {};

template <class... _Tp> struct __tuple_like<__tuple_types<_Tp...> > : true_type {};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TUPLE_TUPLE_LIKE_H
