//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_PREDICATE_H
#define _LIBCUDACXX___CONCEPTS_PREDICATE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__concepts/boolean_testable.h"
#include "../__concepts/invocable.h"
#include "../__functional/invoke.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

template<class _Fn, class... _Args>
concept predicate =
  regular_invocable<_Fn, _Args...> && __boolean_testable<invoke_result_t<_Fn, _Args...>>;

#elif _LIBCUDACXX_STD_VER > 11

// [concept.predicate]
template<class _Fn, class... _Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  _Predicate_,
  requires()
  (requires(regular_invocable<_Fn, _Args...>),
   requires(__boolean_testable<invoke_result_t<_Fn, _Args...>>)));

template<class _Fn, class... _Args>
_LIBCUDACXX_CONCEPT predicate = _LIBCUDACXX_FRAGMENT(_Predicate_, _Fn, _Args...);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_PREDICATE_H
