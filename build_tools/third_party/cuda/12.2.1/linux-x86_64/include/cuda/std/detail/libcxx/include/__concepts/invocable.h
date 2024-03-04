//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_INVOCABLE_H
#define _LIBCUDACXX___CONCEPTS_INVOCABLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__functional/invoke.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.invocable]

template<class _Fn, class... _Args>
concept invocable = requires(_Fn&& __fn, _Args&&... __args) {
  _CUDA_VSTD::__invoke(_CUDA_VSTD::forward<_Fn>(__fn), _CUDA_VSTD::forward<_Args>(__args)...); // not required to be equality preserving
};

// [concept.regular.invocable]

template<class _Fn, class... _Args>
concept regular_invocable = invocable<_Fn, _Args...>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Fn, class... _Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  _Invocable_,
  requires(_Fn&& __fn, _Args&&... __args) //
  (_CUDA_VSTD::__invoke(_CUDA_VSTD::forward<_Fn>(__fn), _CUDA_VSTD::forward<_Args>(__args)...)));

template<class _Fn, class... _Args>
_LIBCUDACXX_CONCEPT invocable = _LIBCUDACXX_FRAGMENT(_Invocable_, _Fn, _Args...);

template<class _Fn, class... _Args>
_LIBCUDACXX_CONCEPT regular_invocable = invocable<_Fn, _Args...>;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_INVOCABLE_H
