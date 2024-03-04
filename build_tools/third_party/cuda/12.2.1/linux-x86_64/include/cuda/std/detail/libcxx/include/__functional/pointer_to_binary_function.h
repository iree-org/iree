// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_POINTER_TO_BINARY_FUNCTION_H
#define _LIBCUDACXX___FUNCTIONAL_POINTER_TO_BINARY_FUNCTION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/binary_function.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

template <class _Arg1, class _Arg2, class _Result>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX11 pointer_to_binary_function
    : public __binary_function<_Arg1, _Arg2, _Result>
{
    _Result (*__f_)(_Arg1, _Arg2);
public:
    _LIBCUDACXX_INLINE_VISIBILITY explicit pointer_to_binary_function(_Result (*__f)(_Arg1, _Arg2))
        : __f_(__f) {}
    _LIBCUDACXX_INLINE_VISIBILITY _Result operator()(_Arg1 __x, _Arg2 __y) const
        {return __f_(__x, __y);}
};

template <class _Arg1, class _Arg2, class _Result>
_LIBCUDACXX_DEPRECATED_IN_CXX11 inline _LIBCUDACXX_INLINE_VISIBILITY
pointer_to_binary_function<_Arg1,_Arg2,_Result>
ptr_fun(_Result (*__f)(_Arg1,_Arg2))
    {return pointer_to_binary_function<_Arg1,_Arg2,_Result>(__f);}

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_POINTER_TO_BINARY_FUNCTION_H
