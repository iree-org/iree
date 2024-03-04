//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_EXCHANGE_H
#define _LIBCUDACXX___UTILITY_EXCHANGE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/is_nothrow_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11
template<class _T1, class _T2 = _T1>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
_T1 exchange(_T1& __obj, _T2&& __new_value)
    noexcept(is_nothrow_move_constructible<_T1>::value && is_nothrow_assignable<_T1&, _T2>::value)
{
    _T1 __old_value = _CUDA_VSTD::move(__obj);
    __obj = _CUDA_VSTD::forward<_T2>(__new_value);
    return __old_value;
}
#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_EXCHANGE_H
