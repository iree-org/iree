//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONVERTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONVERTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/conjunction.h"
#include "../__type_traits/disjunction.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/lazy.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY static void __test_noexcept(_Tp) noexcept;

template<typename _Fm, typename _To>
_LIBCUDACXX_INLINE_VISIBILITY static bool_constant<noexcept(_CUDA_VSTD::__test_noexcept<_To>(_CUDA_VSTD::declval<_Fm>()))>
__is_nothrow_convertible_test();

template <typename _Fm, typename _To>
struct __is_nothrow_convertible_helper: decltype(__is_nothrow_convertible_test<_Fm, _To>())
{ };

template <typename _Fm, typename _To>
struct is_nothrow_convertible : _Or<
    _And<is_void<_To>, is_void<_Fm>>,
    _Lazy<_And, is_convertible<_Fm, _To>, __is_nothrow_convertible_helper<_Fm, _To>>
>::type { };

template <typename _Fm, typename _To>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_convertible_v = is_nothrow_convertible<_Fm, _To>::value;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONVERTIBLE_H
