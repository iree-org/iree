//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_COMMON_REFERENCE_WITH_H
#define _LIBCUDACXX___CONCEPTS_COMMON_REFERENCE_WITH_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/same_as.h"
#include "../__type_traits/common_reference.h"
#include "../__type_traits/copy_cv.h"
#include "../__type_traits/copy_cvref.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.commonref]

template<class _Tp, class _Up>
concept common_reference_with =
  same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>> &&
  convertible_to<_Tp, common_reference_t<_Tp, _Up>> &&
  convertible_to<_Up, common_reference_t<_Tp, _Up>>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __common_reference_exists_,
  requires()(
    typename(common_reference_t<_Tp, _Up>),
    typename(common_reference_t<_Up, _Tp>)
  ));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT _Common_reference_exists = _LIBCUDACXX_FRAGMENT(__common_reference_exists_, _Tp, _Up);

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __common_reference_with_,
  requires()(
    requires(_Common_reference_exists<_Tp, _Up>),
    requires(same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>>),
    requires(convertible_to<_Tp, common_reference_t<_Tp, _Up>>),
    requires(convertible_to<_Up, common_reference_t<_Tp, _Up>>)
  ));

template<class _Tp, class _Up>
_LIBCUDACXX_CONCEPT common_reference_with = _LIBCUDACXX_FRAGMENT(__common_reference_with_, _Tp, _Up);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_COMMON_REFERENCE_WITH_H
