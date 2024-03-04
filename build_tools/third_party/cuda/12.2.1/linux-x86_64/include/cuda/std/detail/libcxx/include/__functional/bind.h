// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_BIND_H
#define _LIBCUDACXX___FUNCTIONAL_BIND_H

// `cuda::std::bind` is not currently supported.

#ifndef __cuda_std__

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/invoke.h"
#include "../__functional/reference_wrapper.h"
#include "../__functional/weak_result_type.h"
#include "../__fwd/get.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_indices.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/forward.h"

#include "../cstddef"
#include "../tuple"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template<class _Tp>
struct is_bind_expression : _If<
    _IsSame<_Tp, __remove_cvref_t<_Tp> >::value,
    false_type,
    is_bind_expression<__remove_cvref_t<_Tp> >
> {};

#if _LIBCUDACXX_STD_VER > 14
template <class _Tp>
inline constexpr size_t is_bind_expression_v = is_bind_expression<_Tp>::value;
#endif

template<class _Tp>
struct is_placeholder : _If<
    _IsSame<_Tp, __remove_cvref_t<_Tp> >::value,
    integral_constant<int, 0>,
    is_placeholder<__remove_cvref_t<_Tp> >
> {};

#if _LIBCUDACXX_STD_VER > 14
template <class _Tp>
inline constexpr size_t is_placeholder_v = is_placeholder<_Tp>::value;
#endif

namespace placeholders
{

template <int _Np> struct __ph {};

#if defined(_LIBCUDACXX_CXX03_LANG) || defined(_LIBCUDACXX_BUILDING_LIBRARY)
_LIBCUDACXX_FUNC_VIS extern const __ph<1>   _1;
_LIBCUDACXX_FUNC_VIS extern const __ph<2>   _2;
_LIBCUDACXX_FUNC_VIS extern const __ph<3>   _3;
_LIBCUDACXX_FUNC_VIS extern const __ph<4>   _4;
_LIBCUDACXX_FUNC_VIS extern const __ph<5>   _5;
_LIBCUDACXX_FUNC_VIS extern const __ph<6>   _6;
_LIBCUDACXX_FUNC_VIS extern const __ph<7>   _7;
_LIBCUDACXX_FUNC_VIS extern const __ph<8>   _8;
_LIBCUDACXX_FUNC_VIS extern const __ph<9>   _9;
_LIBCUDACXX_FUNC_VIS extern const __ph<10> _10;
#else
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<1>   _1{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<2>   _2{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<3>   _3{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<4>   _4{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<5>   _5{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<6>   _6{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<7>   _7{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<8>   _8{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<9>   _9{};
/* _LIBCUDACXX_INLINE_VAR */ constexpr __ph<10> _10{};
#endif // defined(_LIBCUDACXX_CXX03_LANG) || defined(_LIBCUDACXX_BUILDING_LIBRARY)

} // namespace placeholders

template<int _Np>
struct is_placeholder<placeholders::__ph<_Np> >
    : public integral_constant<int, _Np> {};


#ifndef _LIBCUDACXX_CXX03_LANG

template <class _Tp, class _Uj>
inline _LIBCUDACXX_INLINE_VISIBILITY
_Tp&
__mu(reference_wrapper<_Tp> __t, _Uj&)
{
    return __t.get();
}

template <class _Ti, class ..._Uj, size_t ..._Indx>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename __invoke_of<_Ti&, _Uj...>::type
__mu_expand(_Ti& __ti, tuple<_Uj...>& __uj, __tuple_indices<_Indx...>)
{
    return __ti(_CUDA_VSTD::forward<_Uj>(_CUDA_VSTD::get<_Indx>(__uj))...);
}

template <class _Ti, class ..._Uj>
inline _LIBCUDACXX_INLINE_VISIBILITY
__enable_if_t
<
    is_bind_expression<_Ti>::value,
    __invoke_of<_Ti&, _Uj...>
>
__mu(_Ti& __ti, tuple<_Uj...>& __uj)
{
    typedef typename __make_tuple_indices<sizeof...(_Uj)>::type __indices;
    return _CUDA_VSTD::__mu_expand(__ti, __uj, __indices());
}

template <bool IsPh, class _Ti, class _Uj>
struct __mu_return2 {};

template <class _Ti, class _Uj>
struct __mu_return2<true, _Ti, _Uj>
{
    typedef typename tuple_element<is_placeholder<_Ti>::value - 1, _Uj>::type type;
};

template <class _Ti, class _Uj>
inline _LIBCUDACXX_INLINE_VISIBILITY
__enable_if_t
<
    0 < is_placeholder<_Ti>::value,
    typename __mu_return2<0 < is_placeholder<_Ti>::value, _Ti, _Uj>::type
>
__mu(_Ti&, _Uj& __uj)
{
    const size_t _Indx = is_placeholder<_Ti>::value - 1;
    return _CUDA_VSTD::forward<typename tuple_element<_Indx, _Uj>::type>(_CUDA_VSTD::get<_Indx>(__uj));
}

template <class _Ti, class _Uj>
inline _LIBCUDACXX_INLINE_VISIBILITY
__enable_if_t
<
    !is_bind_expression<_Ti>::value &&
    is_placeholder<_Ti>::value == 0 &&
    !__is_reference_wrapper<_Ti>::value,
    _Ti&
>
__mu(_Ti& __ti, _Uj&)
{
    return __ti;
}

template <class _Ti, bool IsReferenceWrapper, bool IsBindEx, bool IsPh,
          class _TupleUj>
struct __mu_return_impl;

template <bool _Invokable, class _Ti, class ..._Uj>
struct __mu_return_invokable  // false
{
    typedef __nat type;
};

template <class _Ti, class ..._Uj>
struct __mu_return_invokable<true, _Ti, _Uj...>
{
    typedef typename __invoke_of<_Ti&, _Uj...>::type type;
};

template <class _Ti, class ..._Uj>
struct __mu_return_impl<_Ti, false, true, false, tuple<_Uj...> >
    : public __mu_return_invokable<__invokable<_Ti&, _Uj...>::value, _Ti, _Uj...>
{
};

template <class _Ti, class _TupleUj>
struct __mu_return_impl<_Ti, false, false, true, _TupleUj>
{
    typedef typename tuple_element<is_placeholder<_Ti>::value - 1,
                                   _TupleUj>::type&& type;
};

template <class _Ti, class _TupleUj>
struct __mu_return_impl<_Ti, true, false, false, _TupleUj>
{
    typedef typename _Ti::type& type;
};

template <class _Ti, class _TupleUj>
struct __mu_return_impl<_Ti, false, false, false, _TupleUj>
{
    typedef _Ti& type;
};

template <class _Ti, class _TupleUj>
struct __mu_return
    : public __mu_return_impl<_Ti,
                              __is_reference_wrapper<_Ti>::value,
                              is_bind_expression<_Ti>::value,
                              0 < is_placeholder<_Ti>::value &&
                              is_placeholder<_Ti>::value <= tuple_size<_TupleUj>::value,
                              _TupleUj>
{
};

template <class _Fp, class _BoundArgs, class _TupleUj>
struct __is_valid_bind_return
{
    static const bool value = false;
};

template <class _Fp, class ..._BoundArgs, class _TupleUj>
struct __is_valid_bind_return<_Fp, tuple<_BoundArgs...>, _TupleUj>
{
    static const bool value = __invokable<_Fp,
                    typename __mu_return<_BoundArgs, _TupleUj>::type...>::value;
};

template <class _Fp, class ..._BoundArgs, class _TupleUj>
struct __is_valid_bind_return<_Fp, const tuple<_BoundArgs...>, _TupleUj>
{
    static const bool value = __invokable<_Fp,
                    typename __mu_return<const _BoundArgs, _TupleUj>::type...>::value;
};

template <class _Fp, class _BoundArgs, class _TupleUj,
          bool = __is_valid_bind_return<_Fp, _BoundArgs, _TupleUj>::value>
struct __bind_return;

template <class _Fp, class ..._BoundArgs, class _TupleUj>
struct __bind_return<_Fp, tuple<_BoundArgs...>, _TupleUj, true>
{
    typedef typename __invoke_of
    <
        _Fp&,
        typename __mu_return
        <
            _BoundArgs,
            _TupleUj
        >::type...
    >::type type;
};

template <class _Fp, class ..._BoundArgs, class _TupleUj>
struct __bind_return<_Fp, const tuple<_BoundArgs...>, _TupleUj, true>
{
    typedef typename __invoke_of
    <
        _Fp&,
        typename __mu_return
        <
            const _BoundArgs,
            _TupleUj
        >::type...
    >::type type;
};

template <class _Fp, class _BoundArgs, class _TupleUj>
using __bind_return_t = typename __bind_return<_Fp, _BoundArgs, _TupleUj>::type;

template <class _Fp, class _BoundArgs, size_t ..._Indx, class _Args>
inline _LIBCUDACXX_INLINE_VISIBILITY
__bind_return_t<_Fp, _BoundArgs, _Args>
__apply_functor(_Fp& __f, _BoundArgs& __bound_args, __tuple_indices<_Indx...>,
                _Args&& __args)
{
    return _CUDA_VSTD::__invoke(__f, _CUDA_VSTD::__mu(_CUDA_VSTD::get<_Indx>(__bound_args), __args)...);
}

template<class _Fp, class ..._BoundArgs>
class __bind : public __weak_result_type<__decay_t<_Fp>>
{
protected:
    typedef __decay_t<_Fp> _Fd;
    typedef tuple<__decay_t<_BoundArgs>...> _Td;
private:
    _Fd __f_;
    _Td __bound_args_;

    typedef typename __make_tuple_indices<sizeof...(_BoundArgs)>::type __indices;
public:
    template <class _Gp, class ..._BA,
              class = __enable_if_t
                               <
                                  is_constructible<_Fd, _Gp>::value &&
                                  !is_same<__libcpp_remove_reference_t<_Gp>,
                                           __bind>::value
                               >>
      _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
      explicit __bind(_Gp&& __f, _BA&& ...__bound_args)
        : __f_(_CUDA_VSTD::forward<_Gp>(__f)),
          __bound_args_(_CUDA_VSTD::forward<_BA>(__bound_args)...) {}

    template <class ..._Args>
        _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
        __bind_return_t<_Fd, _Td, tuple<_Args&&...>>
        operator()(_Args&& ...__args)
        {
            return _CUDA_VSTD::__apply_functor(__f_, __bound_args_, __indices(),
                                  tuple<_Args&&...>(_CUDA_VSTD::forward<_Args>(__args)...));
        }

    template <class ..._Args>
        _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
        __bind_return_t<const _Fd, const _Td, tuple<_Args&&...>>
        operator()(_Args&& ...__args) const
        {
            return _CUDA_VSTD::__apply_functor(__f_, __bound_args_, __indices(),
                                   tuple<_Args&&...>(_CUDA_VSTD::forward<_Args>(__args)...));
        }
};

template<class _Fp, class ..._BoundArgs>
struct is_bind_expression<__bind<_Fp, _BoundArgs...> > : public true_type {};

template<class _Rp, class _Fp, class ..._BoundArgs>
class __bind_r
    : public __bind<_Fp, _BoundArgs...>
{
    typedef __bind<_Fp, _BoundArgs...> base;
    typedef typename base::_Fd _Fd;
    typedef typename base::_Td _Td;
public:
    typedef _Rp result_type;


    template <class _Gp, class ..._BA,
              class = __enable_if_t
                               <
                                  is_constructible<_Fd, _Gp>::value &&
                                  !is_same<__libcpp_remove_reference_t<_Gp>,
                                           __bind_r>::value
                               >>
      _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
      explicit __bind_r(_Gp&& __f, _BA&& ...__bound_args)
        : base(_CUDA_VSTD::forward<_Gp>(__f),
               _CUDA_VSTD::forward<_BA>(__bound_args)...) {}

    template <class ..._Args>
        _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
        __enable_if_t
        <
            is_convertible<__bind_return_t<_Fd, _Td, tuple<_Args&&...>>,
                           result_type>::value || is_void<_Rp>::value,
            result_type
        >
        operator()(_Args&& ...__args)
        {
            typedef __invoke_void_return_wrapper<_Rp> _Invoker;
            return _Invoker::__call(static_cast<base&>(*this), _CUDA_VSTD::forward<_Args>(__args)...);
        }

    template <class ..._Args>
        _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
        __enable_if_t
        <
            is_convertible<__bind_return_t<const _Fd, const _Td, tuple<_Args&&...>>,
                           result_type>::value || is_void<_Rp>::value,
            result_type
        >
        operator()(_Args&& ...__args) const
        {
            typedef __invoke_void_return_wrapper<_Rp> _Invoker;
            return _Invoker::__call(static_cast<base const&>(*this), _CUDA_VSTD::forward<_Args>(__args)...);
        }
};

template<class _Rp, class _Fp, class ..._BoundArgs>
struct is_bind_expression<__bind_r<_Rp, _Fp, _BoundArgs...> > : public true_type {};

template<class _Fp, class ..._BoundArgs>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
__bind<_Fp, _BoundArgs...>
bind(_Fp&& __f, _BoundArgs&&... __bound_args)
{
    typedef __bind<_Fp, _BoundArgs...> type;
    return type(_CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_BoundArgs>(__bound_args)...);
}

template<class _Rp, class _Fp, class ..._BoundArgs>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
__bind_r<_Rp, _Fp, _BoundArgs...>
bind(_Fp&& __f, _BoundArgs&&... __bound_args)
{
    typedef __bind_r<_Rp, _Fp, _BoundArgs...> type;
    return type(_CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_BoundArgs>(__bound_args)...);
}

#endif // _LIBCUDACXX_CXX03_LANG

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __cuda_std__

#endif // _LIBCUDACXX___FUNCTIONAL_BIND_H
