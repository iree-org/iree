// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_NOT_FN_H
#define _LIBCUDACXX___FUNCTIONAL_NOT_FN_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/invoke.h"
#include "../__functional/perfect_forward.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

struct __not_fn_op {
    template <class... _Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 auto operator()(_Args&&... __args) const
        noexcept(noexcept(!_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...)))
        -> decltype(      !_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...))
        { return          !_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Args>(__args)...); }
};

template <class _Fn>
struct __not_fn_t : __perfect_forward<__not_fn_op, _Fn> {
    using __perfect_forward<__not_fn_op, _Fn>::__perfect_forward;
};

template <class _Fn, class = enable_if_t<
    is_constructible_v<decay_t<_Fn>, _Fn> &&
    is_move_constructible_v<decay_t<_Fn>>
>>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_CONSTEXPR_AFTER_CXX17 auto not_fn(_Fn&& __f) {
    return __not_fn_t<decay_t<_Fn>>(_CUDA_VSTD::forward<_Fn>(__f));
}

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_NOT_FN_H
