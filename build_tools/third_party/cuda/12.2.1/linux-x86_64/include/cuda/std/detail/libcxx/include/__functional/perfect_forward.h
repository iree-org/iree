// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_PERFECT_FORWARD_H
#define _LIBCUDACXX___FUNCTIONAL_PERFECT_FORWARD_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__functional/invoke.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"
#include "../__utility/integer_sequence.h"
#include "../__utility/move.h"

#include "../tuple"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

template <class _Op, class _Indices, class... _BoundArgs>
struct __perfect_forward_impl;

template <class _Op, size_t... _Idx, class... _BoundArgs>
struct __perfect_forward_impl<_Op, index_sequence<_Idx...>, _BoundArgs...> {
private:
  tuple<_BoundArgs...> __bound_args_;

public:
  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires is_constructible_v<tuple<_BoundArgs...>, _Args&&...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  explicit constexpr __perfect_forward_impl(_Args&&... __bound_args)
    noexcept(is_nothrow_constructible_v<tuple<_BoundArgs...>, _Args&&...>)
    : __bound_args_(_CUDA_VSTD::forward<_Args>(__bound_args)...) {}

  __perfect_forward_impl(__perfect_forward_impl const&) = default;
  __perfect_forward_impl(__perfect_forward_impl&&) = default;

  __perfect_forward_impl& operator=(__perfect_forward_impl const&) = default;
  __perfect_forward_impl& operator=(__perfect_forward_impl&&) = default;

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires is_invocable_v<_Op, _BoundArgs&..., _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Args&&... __args) &
    noexcept(noexcept(_Op()(_CUDA_VSTD::get<_Idx>(__bound_args_)..., _CUDA_VSTD::forward<_Args>(__args)...)))
    -> decltype(      _Op()(_CUDA_VSTD::get<_Idx>(__bound_args_)..., _CUDA_VSTD::forward<_Args>(__args)...))
    { return          _Op()(_CUDA_VSTD::get<_Idx>(__bound_args_)..., _CUDA_VSTD::forward<_Args>(__args)...); }

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires (!is_invocable_v<_Op, _BoundArgs&..., _Args...>))
  _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Args&&...) & = delete;

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires is_invocable_v<_Op, _BoundArgs const&..., _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Args&&... __args) const&
    noexcept(noexcept(_Op()(_CUDA_VSTD::get<_Idx>(__bound_args_)..., _CUDA_VSTD::forward<_Args>(__args)...)))
    -> decltype(      _Op()(_CUDA_VSTD::get<_Idx>(__bound_args_)..., _CUDA_VSTD::forward<_Args>(__args)...))
    { return          _Op()(_CUDA_VSTD::get<_Idx>(__bound_args_)..., _CUDA_VSTD::forward<_Args>(__args)...); }

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires (!is_invocable_v<_Op, _BoundArgs const&..., _Args...>))
  _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Args&&...) const& = delete;

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires is_invocable_v<_Op, _BoundArgs..., _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Args&&... __args) &&
    noexcept(noexcept(_Op()(_CUDA_VSTD::get<_Idx>(_CUDA_VSTD::move(__bound_args_))..., _CUDA_VSTD::forward<_Args>(__args)...)))
    -> decltype(      _Op()(_CUDA_VSTD::get<_Idx>(_CUDA_VSTD::move(__bound_args_))..., _CUDA_VSTD::forward<_Args>(__args)...))
    { return          _Op()(_CUDA_VSTD::get<_Idx>(_CUDA_VSTD::move(__bound_args_))..., _CUDA_VSTD::forward<_Args>(__args)...); }

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires (!is_invocable_v<_Op, _BoundArgs..., _Args...>))
  _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Args&&...) && = delete;

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires is_invocable_v<_Op, _BoundArgs const..., _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(_Args&&... __args) const&&
    noexcept(noexcept(_Op()(_CUDA_VSTD::get<_Idx>(_CUDA_VSTD::move(__bound_args_))..., _CUDA_VSTD::forward<_Args>(__args)...)))
    -> decltype(      _Op()(_CUDA_VSTD::get<_Idx>(_CUDA_VSTD::move(__bound_args_))..., _CUDA_VSTD::forward<_Args>(__args)...))
    { return          _Op()(_CUDA_VSTD::get<_Idx>(_CUDA_VSTD::move(__bound_args_))..., _CUDA_VSTD::forward<_Args>(__args)...); }

  _LIBCUDACXX_TEMPLATE(class... _Args)
    (requires (!is_invocable_v<_Op, _BoundArgs const..., _Args...>))
  _LIBCUDACXX_INLINE_VISIBILITY auto operator()(_Args&&...) const&& = delete;
};

// __perfect_forward implements a perfect-forwarding call wrapper as explained in [func.require].
template <class _Op, class ..._Args>
using __perfect_forward = __perfect_forward_impl<_Op, index_sequence_for<_Args...>, _Args...>;

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_PERFECT_FORWARD_H
