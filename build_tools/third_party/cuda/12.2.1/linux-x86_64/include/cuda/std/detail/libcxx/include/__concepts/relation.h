//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_RELATION_H
#define _LIBCUDACXX___CONCEPTS_RELATION_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/predicate.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.relation]

template<class _Rp, class _Tp, class _Up>
concept relation =
  predicate<_Rp, _Tp, _Tp> && predicate<_Rp, _Up, _Up> &&
  predicate<_Rp, _Tp, _Up> && predicate<_Rp, _Up, _Tp>;

// [concept.equiv]

template<class _Rp, class _Tp, class _Up>
concept equivalence_relation = relation<_Rp, _Tp, _Up>;

// [concept.strictweakorder]

template<class _Rp, class _Tp, class _Up>
concept strict_weak_order = relation<_Rp, _Tp, _Up>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Rp, class _Tp, class _Up>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __relation_,
  requires()(
    requires(predicate<_Rp, _Tp, _Tp>),
    requires(predicate<_Rp, _Up, _Up>),
    requires(predicate<_Rp, _Tp, _Up>),
    requires(predicate<_Rp, _Up, _Tp>)
  ));

template<class _Rp, class _Tp, class _Up>
_LIBCUDACXX_CONCEPT relation = _LIBCUDACXX_FRAGMENT(__relation_, _Rp, _Tp, _Up);

// [concept.equiv]

template<class _Rp, class _Tp, class _Up>
_LIBCUDACXX_CONCEPT equivalence_relation = relation<_Rp, _Tp, _Up>;

// [concept.strictweakorder]

template<class _Rp, class _Tp, class _Up>
_LIBCUDACXX_CONCEPT strict_weak_order = relation<_Rp, _Tp, _Up>;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_RELATION_H
