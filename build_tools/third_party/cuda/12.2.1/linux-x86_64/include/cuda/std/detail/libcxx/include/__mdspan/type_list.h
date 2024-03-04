/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _LIBCUDACXX___MDSPAN_TYPE_LIST_HPP
#define _LIBCUDACXX___MDSPAN_TYPE_LIST_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/macros.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

//==============================================================================

namespace __detail {

template <class... _Ts> struct __type_list { static constexpr auto __size = sizeof...(_Ts); };

// Implementation of type_list at() that's heavily optimized for small typelists
template <size_t, class> struct __type_at;
template <size_t, class _Seq, class=_CUDA_VSTD::make_index_sequence<_Seq::__size>> struct __type_at_large_impl;

template <size_t _Ip, size_t _Idx, class _Tp>
struct __type_at_entry { };

template <class _Result>
struct __type_at_assign_op_ignore_rest {
  template <class _Tp>
  __MDSPAN_HOST_DEVICE
  __type_at_assign_op_ignore_rest<_Result> operator=(_Tp&&);
  using type = _Result;
};

struct __type_at_assign_op_impl {
  template <size_t _Ip, size_t _Idx, class _Tp>
  __MDSPAN_HOST_DEVICE
  __type_at_assign_op_impl operator=(__type_at_entry<_Ip, _Idx, _Tp>&&);
  template <size_t _Ip, class _Tp>
  __MDSPAN_HOST_DEVICE
  __type_at_assign_op_ignore_rest<_Tp> operator=(__type_at_entry<_Ip, _Ip, _Tp>&&);
};

template <size_t _Ip, class... _Ts, size_t... _Idxs>
struct __type_at_large_impl<_Ip, __type_list<_Ts...>, _CUDA_VSTD::integer_sequence<size_t, _Idxs...>>
  : decltype(
      __MDSPAN_FOLD_ASSIGN_LEFT(__type_at_assign_op_impl{}, /* = ... = */ __type_at_entry<_Ip, _Idxs, _Ts>{})
    )
{ };

template <size_t _Ip, class... _Ts>
struct __type_at<_Ip, __type_list<_Ts...>>
    : __type_at_large_impl<_Ip, __type_list<_Ts...>>
{ };

template <class _T0, class... _Ts>
struct __type_at<0, __type_list<_T0, _Ts...>> {
  using type = _T0;
};

template <class _T0, class _T1, class... _Ts>
struct __type_at<1, __type_list<_T0, _T1, _Ts...>> {
  using type = _T1;
};

template <class _T0, class _T1, class _T2, class... _Ts>
struct __type_at<2, __type_list<_T0, _T1, _T2, _Ts...>> {
  using type = _T2;
};

template <class _T0, class _T1, class _T2, class _T3, class... _Ts>
struct __type_at<3, __type_list<_T0, _T1, _T2, _T3, _Ts...>> {
  using type = _T3;
};


} // namespace __detail

//==============================================================================

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_TYPE_LIST_HPP
