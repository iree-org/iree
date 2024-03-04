//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ASSIGNABLE_H
#define _LIBCUDACXX___CONCEPTS_ASSIGNABLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/common_reference_with.h"
#include "../__concepts/same_as.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/make_const_lvalue_ref.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.assignable]

template<class _Lhs, class _Rhs>
concept assignable_from =
  is_lvalue_reference_v<_Lhs> &&
  common_reference_with<__make_const_lvalue_ref<_Lhs>, __make_const_lvalue_ref<_Rhs>> &&
  requires (_Lhs __lhs, _Rhs&& __rhs) {
    { __lhs = _CUDA_VSTD::forward<_Rhs>(__rhs) } -> same_as<_Lhs>;
  };

#elif _LIBCUDACXX_STD_VER > 11

template<class _Lhs, class _Rhs>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __assignable_from_,
  requires(_Lhs __lhs, _Rhs&& __rhs)(
    requires(_LIBCUDACXX_TRAIT(is_lvalue_reference, _Lhs)),
    requires(common_reference_with<__make_const_lvalue_ref<_Lhs>, __make_const_lvalue_ref<_Rhs>>),
    requires(same_as<_Lhs, decltype(__lhs = _CUDA_VSTD::forward<_Rhs>(__rhs))>)
  ));

template<class _Lhs, class _Rhs>
_LIBCUDACXX_CONCEPT assignable_from = _LIBCUDACXX_FRAGMENT(__assignable_from_, _Lhs, _Rhs);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_ASSIGNABLE_H
