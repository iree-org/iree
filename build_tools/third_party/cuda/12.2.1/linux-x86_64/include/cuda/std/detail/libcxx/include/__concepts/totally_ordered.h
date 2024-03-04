//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_TOTALLY_ORDERED_H
#define _LIBCUDACXX___CONCEPTS_TOTALLY_ORDERED_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/boolean_testable.h"
#include "../__concepts/equality_comparable.h"
#include "../__type_traits/common_reference.h"
#include "../__type_traits/make_const_lvalue_ref.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.totallyordered]

template<class _Tp, class _Up>
concept __partially_ordered_with =
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) {
    { __t <  __u } -> __boolean_testable;
    { __t >  __u } -> __boolean_testable;
    { __t <= __u } -> __boolean_testable;
    { __t >= __u } -> __boolean_testable;
    { __u <  __t } -> __boolean_testable;
    { __u >  __t } -> __boolean_testable;
    { __u <= __t } -> __boolean_testable;
    { __u >= __t } -> __boolean_testable;
  };

template<class _Tp>
concept totally_ordered = equality_comparable<_Tp> && __partially_ordered_with<_Tp, _Tp>;

template<class _Tp, class _Up>
concept totally_ordered_with =
  totally_ordered<_Tp> && totally_ordered<_Up> &&
  equality_comparable_with<_Tp, _Up> &&
  totally_ordered<
    common_reference_t<
      __make_const_lvalue_ref<_Tp>,
      __make_const_lvalue_ref<_Up>>> &&
  __partially_ordered_with<_Tp, _Up>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __partially_ordered_with_,
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) //
  (requires(__boolean_testable<decltype(__t <  __u)>),
   requires(__boolean_testable<decltype(__t >  __u)>),
   requires(__boolean_testable<decltype(__t <= __u)>),
   requires(__boolean_testable<decltype(__t >= __u)>),
   requires(__boolean_testable<decltype(__u <  __t)>),
   requires(__boolean_testable<decltype(__u >  __t)>),
   requires(__boolean_testable<decltype(__u <= __t)>),
   requires(__boolean_testable<decltype(__u >= __t)>)));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT __partially_ordered_with = _LIBCUDACXX_FRAGMENT(__partially_ordered_with_, _Tp, _Up);

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __totally_ordered_,
  requires()(
    requires(equality_comparable<_Tp>),
    requires(__partially_ordered_with<_Tp, _Tp>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT totally_ordered = _LIBCUDACXX_FRAGMENT(__totally_ordered_, _Tp);

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __totally_ordered_with_,
  requires()(
    requires(totally_ordered<_Tp>),
    requires(totally_ordered<_Up>),
    requires(equality_comparable_with<_Tp, _Up>),
    requires(totally_ordered<
    common_reference_t<
      __make_const_lvalue_ref<_Tp>,
      __make_const_lvalue_ref<_Up>>>),
    requires(__partially_ordered_with<_Tp, _Up>)));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT totally_ordered_with = _LIBCUDACXX_FRAGMENT(__totally_ordered_with_, _Tp, _Up);;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_TOTALLY_ORDERED_H
