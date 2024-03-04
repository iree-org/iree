// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BINDER2ND_H
#define _LIBCUDACXX___FUNCTIONAL_BINDER2ND_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/unary_function.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

template <class __Operation>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX11 binder2nd
    : public __unary_function<typename __Operation::first_argument_type, typename __Operation::result_type>
{
protected:
    __Operation                                op;
    typename __Operation::second_argument_type value;
public:
    _LIBCUDACXX_INLINE_VISIBILITY
    binder2nd(const __Operation& __x, const typename __Operation::second_argument_type __y)
        : op(__x), value(__y) {}
    _LIBCUDACXX_INLINE_VISIBILITY typename __Operation::result_type operator()
        (      typename __Operation::first_argument_type& __x) const
            {return op(__x, value);}
    _LIBCUDACXX_INLINE_VISIBILITY typename __Operation::result_type operator()
        (const typename __Operation::first_argument_type& __x) const
            {return op(__x, value);}
};

template <class __Operation, class _Tp>
_LIBCUDACXX_DEPRECATED_IN_CXX11 inline _LIBCUDACXX_INLINE_VISIBILITY
binder2nd<__Operation>
bind2nd(const __Operation& __op, const _Tp& __x)
    {return binder2nd<__Operation>(__op, __x);}

#endif // _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_BINDER2ND_H
