// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BINARY_FUNCTION_H
#define _LIBCUDACXX___FUNCTIONAL_BINARY_FUNCTION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg1, class _Arg2, class _Result>
struct _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX11 binary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Result result_type;
};

#endif // _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg1, class _Arg2, class _Result> struct __binary_function_keep_layout_base {
#if _LIBCUDACXX_STD_VER <= 17 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using first_argument_type _LIBCUDACXX_DEPRECATED_IN_CXX17 = _Arg1;
  using second_argument_type _LIBCUDACXX_DEPRECATED_IN_CXX17 = _Arg2;
  using result_type _LIBCUDACXX_DEPRECATED_IN_CXX17 = _Result;
#endif
};

#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
_LIBCUDACXX_DIAGNOSTIC_PUSH
_LIBCUDACXX_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
template <class _Arg1, class _Arg2, class _Result>
using __binary_function = binary_function<_Arg1, _Arg2, _Result>;
_LIBCUDACXX_DIAGNOSTIC_POP
#else
template <class _Arg1, class _Arg2, class _Result>
using __binary_function = __binary_function_keep_layout_base<_Arg1, _Arg2, _Result>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_BINARY_FUNCTION_H
