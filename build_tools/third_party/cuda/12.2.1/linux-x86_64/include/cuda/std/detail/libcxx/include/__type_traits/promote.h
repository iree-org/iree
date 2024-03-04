//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H
#define _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_same.h"
#include "../__utility/declval.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __numeric_type
{
   _LIBCUDACXX_INLINE_VISIBILITY static void __test(...);
   _LIBCUDACXX_INLINE_VISIBILITY static float __test(float);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(char);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(int);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(unsigned);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(long);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(unsigned long);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(long long);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(unsigned long long);
   _LIBCUDACXX_INLINE_VISIBILITY static double __test(double);
   _LIBCUDACXX_INLINE_VISIBILITY static long double __test(long double);

   typedef decltype(__test(declval<_Tp>())) type;
   static const bool value = _IsNotSame<type, void>::value;
};

template <>
struct __numeric_type<void>
{
   static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&
                 __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp
{
public:
    static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
public:
    typedef decltype(__type1() + __type2()) type;
    static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
    typedef typename __numeric_type<_A1>::type type;
    static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H
