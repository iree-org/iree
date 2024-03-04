// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BINARY_NEGATE_H
#define _LIBCUDACXX___FUNCTIONAL_BINARY_NEGATE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/binary_function.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER <= 17 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS)

template <class _Predicate>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX17 binary_negate
    : public __binary_function<typename _Predicate::first_argument_type,
                               typename _Predicate::second_argument_type,
                               bool>
{
    _Predicate __pred_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY explicit _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    binary_negate(const _Predicate& __pred) : __pred_(__pred) {}

    _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _LIBCUDACXX_INLINE_VISIBILITY
    bool operator()(const typename _Predicate::first_argument_type& __x,
                    const typename _Predicate::second_argument_type& __y) const
        {return !__pred_(__x, __y);}
};

template <class _Predicate>
_LIBCUDACXX_DEPRECATED_IN_CXX17 inline _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _LIBCUDACXX_INLINE_VISIBILITY
binary_negate<_Predicate>
not2(const _Predicate& __pred) {return binary_negate<_Predicate>(__pred);}

#endif // _LIBCUDACXX_STD_VER <= 17 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_BINARY_NEGATE_H
