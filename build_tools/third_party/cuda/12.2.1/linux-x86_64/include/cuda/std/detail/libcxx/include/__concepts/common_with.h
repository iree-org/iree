//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_COMMON_WITH_H
#define _LIBCUDACXX___CONCEPTS_COMMON_WITH_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/common_reference_with.h"
#include "../__concepts/same_as.h"
#include "../__type_traits/add_lvalue_reference.h"
#include "../__type_traits/common_type.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.common]

template<class _Tp, class _Up>
concept common_with =
  same_as<common_type_t<_Tp, _Up>, common_type_t<_Up, _Tp>> &&
  requires {
    static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Tp>());
    static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Up>());
  } &&
  common_reference_with<
    add_lvalue_reference_t<const _Tp>,
    add_lvalue_reference_t<const _Up>> &&
  common_reference_with<
    add_lvalue_reference_t<common_type_t<_Tp, _Up>>,
    common_reference_t<
      add_lvalue_reference_t<const _Tp>,
      add_lvalue_reference_t<const _Up>>>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __common_type_exists_,
  requires()(
    typename(common_type_t<_Tp, _Up>),
    typename(common_type_t<_Up, _Tp>)
  ));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT _Common_type_exists = _LIBCUDACXX_FRAGMENT(__common_type_exists_, _Tp, _Up);

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __common_type_constructible_,
  requires()(
    requires(_Common_type_exists<_Tp, _Up>),
    static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Tp>()),
    static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Up>())
  ));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT _Common_type_constructible = _LIBCUDACXX_FRAGMENT(__common_type_constructible_, _Tp, _Up);


template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __common_with_,
  requires()(
    requires(_Common_type_constructible<_Tp, _Up>),
    requires(same_as<common_type_t<_Tp, _Up>, common_type_t<_Up, _Tp>>),
    requires(common_reference_with<
              add_lvalue_reference_t<const _Tp>,
              add_lvalue_reference_t<const _Up>>),
    requires(common_reference_with<
              add_lvalue_reference_t<common_type_t<_Tp, _Up>>,
              common_reference_t<
                add_lvalue_reference_t<const _Tp>,
                add_lvalue_reference_t<const _Up>>>)));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT common_with = _LIBCUDACXX_FRAGMENT(__common_with_, _Tp, _Up);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_COMMON_WITH_H
