//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_CONSTRUCTIBLE_H
#define _LIBCUDACXX___CONCEPTS_CONSTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/destructible.h"
#include "../__type_traits/is_constructible.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.constructible]
template<class _Tp, class... _Args>
concept constructible_from =
    destructible<_Tp> && is_constructible_v<_Tp, _Args...>;

// [concept.default.init]
template<class _Tp>
concept __default_initializable = requires { ::new _Tp; };

template<class _Tp>
concept default_initializable = constructible_from<_Tp> &&
    requires { _Tp{}; } && __default_initializable<_Tp>;

// [concept.moveconstructible]
template<class _Tp>
concept move_constructible =
  constructible_from<_Tp, _Tp> && convertible_to<_Tp, _Tp>;

// [concept.copyconstructible]
template<class _Tp>
concept copy_constructible =
  move_constructible<_Tp> &&
  constructible_from<_Tp, _Tp&> && convertible_to<_Tp&, _Tp> &&
  constructible_from<_Tp, const _Tp&> && convertible_to<const _Tp&, _Tp> &&
  constructible_from<_Tp, const _Tp> && convertible_to<const _Tp, _Tp>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Tp, class... _Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __constructible_from_,
  requires()(
    requires(destructible<_Tp>),
    requires(_LIBCUDACXX_TRAIT(is_constructible, _Tp, _Args...))
  ));

template<class _Tp, class... _Args>
_LIBCUDACXX_CONCEPT constructible_from = _LIBCUDACXX_FRAGMENT(__constructible_from_, _Tp, _Args...);

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __default_initializable_,
  requires()(
    (::new _Tp)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT __default_initializable = _LIBCUDACXX_FRAGMENT(__default_initializable_, _Tp);

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  _Default_initializable_,
  requires(_Tp = _Tp{}) (
    requires(constructible_from<_Tp>),
    requires(__default_initializable<_Tp>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT default_initializable = _LIBCUDACXX_FRAGMENT(_Default_initializable_, _Tp);

// [concept.moveconstructible]
template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __move_constructible_,
  requires()(
    requires(constructible_from<_Tp, _Tp>),
    requires(convertible_to<_Tp, _Tp>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT move_constructible = _LIBCUDACXX_FRAGMENT(__move_constructible_, _Tp);

// [concept.copyconstructible]
template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __copy_constructible_,
  requires()(
    requires(move_constructible<_Tp>),
    requires(constructible_from<_Tp, add_lvalue_reference_t<_Tp>> && convertible_to<add_lvalue_reference_t<_Tp>, _Tp>),
    requires(constructible_from<_Tp, const add_lvalue_reference_t<_Tp>> && convertible_to<const add_lvalue_reference_t<_Tp>, _Tp>),
    requires(constructible_from<_Tp, const _Tp> && convertible_to<const _Tp, _Tp>)
  ));

template<class _Tp>
_LIBCUDACXX_CONCEPT copy_constructible =  _LIBCUDACXX_FRAGMENT(__copy_constructible_, _Tp);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_CONSTRUCTIBLE_H
