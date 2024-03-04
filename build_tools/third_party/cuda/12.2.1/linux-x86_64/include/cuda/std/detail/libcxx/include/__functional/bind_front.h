// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BIND_FRONT_H
#define _LIBCUDACXX___FUNCTIONAL_BIND_FRONT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__functional/invoke.h"
#include "../__functional/perfect_forward.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

struct __bind_front_op {
    template <class ..._Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Args&& ...__args) const
        noexcept(noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...)))
        -> decltype(      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...))
        { return          _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...); }
};

template <class _Fn, class ..._BoundArgs>
struct __bind_front_t : __perfect_forward<__bind_front_op, _Fn, _BoundArgs...> {
    using __perfect_forward<__bind_front_op, _Fn, _BoundArgs...>::__perfect_forward;
};

_LIBCUDACXX_TEMPLATE(class _Fn, class... _Args)
  (requires is_constructible_v<decay_t<_Fn>, _Fn> _LIBCUDACXX_AND
            is_move_constructible_v<decay_t<_Fn>>_LIBCUDACXX_AND
            (is_constructible_v<decay_t<_Args>, _Args> && ...) _LIBCUDACXX_AND
            (is_move_constructible_v<decay_t<_Args>> && ... ))
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
constexpr auto bind_front(_Fn&& __f, _Args&&... __args) noexcept(is_nothrow_constructible_v<tuple<decay_t<_Args>...>, _Args&&...>) {
    return __bind_front_t<decay_t<_Fn>, decay_t<_Args>...>(_CUDA_VSTD::forward<_Fn>(__f), _CUDA_VSTD::forward<_Args>(__args)...);
}

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_BIND_FRONT_H
