//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_EQUALITY_COMPARABLE_H
#define _LIBCUDACXX___CONCEPTS_EQUALITY_COMPARABLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/boolean_testable.h"
#include "../__concepts/common_reference_with.h"
#include "../__type_traits/common_reference.h"
#include "../__type_traits/make_const_lvalue_ref.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.equalitycomparable]

template<class _Tp, class _Up>
concept __weakly_equality_comparable_with =
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) {
    { __t == __u } -> __boolean_testable;
    { __t != __u } -> __boolean_testable;
    { __u == __t } -> __boolean_testable;
    { __u != __t } -> __boolean_testable;
  };

template<class _Tp>
concept equality_comparable = __weakly_equality_comparable_with<_Tp, _Tp>;

template<class _Tp, class _Up>
concept equality_comparable_with =
  equality_comparable<_Tp> && equality_comparable<_Up> &&
  common_reference_with<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>> &&
  equality_comparable<
    common_reference_t<
      __make_const_lvalue_ref<_Tp>,
      __make_const_lvalue_ref<_Up>>> &&
  __weakly_equality_comparable_with<_Tp, _Up>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __with_lvalue_reference_,
  requires()(
    typename(__make_const_lvalue_ref<_Tp>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT _With_lvalue_reference = _LIBCUDACXX_FRAGMENT(__with_lvalue_reference_, _Tp);


template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __weakly_equality_comparable_with_,
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) //
  (requires(_With_lvalue_reference<_Tp>),
   requires(_With_lvalue_reference<_Up>),
   requires(__boolean_testable<decltype(__t == __u)>),
   requires(__boolean_testable<decltype(__t != __u)>),
   requires(__boolean_testable<decltype(__u == __t)>),
   requires(__boolean_testable<decltype(__u != __t)>)));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT __weakly_equality_comparable_with =
  _LIBCUDACXX_FRAGMENT(__weakly_equality_comparable_with_, _Tp, _Up);

template<class _Tp>
_LIBCUDACXX_CONCEPT equality_comparable = __weakly_equality_comparable_with<_Tp, _Tp>;

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __equality_comparable_with_,
  requires()(
    requires(equality_comparable<_Tp>),
    requires(equality_comparable<_Up>),
    requires(common_reference_with<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>),
    requires(equality_comparable<
    common_reference_t<
      __make_const_lvalue_ref<_Tp>,
      __make_const_lvalue_ref<_Up>>>),
    requires(__weakly_equality_comparable_with<_Tp, _Up>)));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT equality_comparable_with = _LIBCUDACXX_FRAGMENT(__equality_comparable_with_, _Tp, _Up);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_EQUALITY_COMPARABLE_H
