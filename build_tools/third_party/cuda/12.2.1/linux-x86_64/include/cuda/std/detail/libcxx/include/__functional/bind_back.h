// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BIND_BACK_H
#define _LIBCUDACXX___FUNCTIONAL_BIND_BACK_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/invoke.h"
#include "../__functional/perfect_forward.h"
#include "../__fwd/get.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__utility/forward.h"
#include "../__utility/integer_sequence.h"

#include "../tuple"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

template <size_t _NBound, class = make_index_sequence<_NBound>>
struct __bind_back_op;

template <size_t _NBound, size_t ..._Ip>
struct __bind_back_op<_NBound, index_sequence<_Ip...>> {
    template <class _Fn, class _BoundArgs, class... _Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Fn&& __f, _BoundArgs&& __bound_args, _Args&&... __args) const
        noexcept(noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)..., _CUDA_VSTD::get<_Ip>(_CUDA_VSTD::forward<_BoundArgs>(__bound_args))...)))
        -> decltype(      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)..., _CUDA_VSTD::get<_Ip>(_CUDA_VSTD::forward<_BoundArgs>(__bound_args))...))
        { return          _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)..., _CUDA_VSTD::get<_Ip>(_CUDA_VSTD::forward<_BoundArgs>(__bound_args))...); }
};

template <class _Fn, class _BoundArgs>
struct __bind_back_t : __perfect_forward<__bind_back_op<tuple_size_v<_BoundArgs>>, _Fn, _BoundArgs> {
    using __perfect_forward<__bind_back_op<tuple_size_v<_BoundArgs>>, _Fn, _BoundArgs>::__perfect_forward;
};

template <class _Fn, class ..._Args, class = enable_if_t<
    _And<
        is_constructible<decay_t<_Fn>, _Fn>,
        is_move_constructible<decay_t<_Fn>>,
        is_constructible<decay_t<_Args>, _Args>...,
        is_move_constructible<decay_t<_Args>>...
    >::value
>>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
constexpr auto __bind_back(_Fn&& __f, _Args&&... __args)
    noexcept(noexcept(__bind_back_t<decay_t<_Fn>, tuple<decay_t<_Args>...>>(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::forward<_Args>(__args)...))))
    -> decltype(      __bind_back_t<decay_t<_Fn>, tuple<decay_t<_Args>...>>(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::forward<_Args>(__args)...)))
    { return          __bind_back_t<decay_t<_Fn>, tuple<decay_t<_Args>...>>(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward_as_tuple(_CUDA_VSTD::forward<_Args>(__args)...)); }

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_BIND_BACK_H
