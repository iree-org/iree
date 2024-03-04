/* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_SCAN_H_
#define _CG_SCAN_H_

#include "info.h"
#include "helpers.h"
#include "functional.h"
#include "coalesced_scan.h"

_CG_BEGIN_NAMESPACE

namespace details {

    // Group support for scan.
    template <class TyGroup> struct _scan_group_supported : public _CG_STL_NAMESPACE::false_type {};

    template <unsigned int Sz, typename TyPar>
    struct _scan_group_supported<cooperative_groups::thread_block_tile<Sz, TyPar>> : public _CG_STL_NAMESPACE::true_type {};
    template <unsigned int Sz, typename TyPar>
    struct _scan_group_supported<internal_thread_block_tile<Sz, TyPar>>            : public _CG_STL_NAMESPACE::true_type {};
    template <>
    struct _scan_group_supported<cooperative_groups::coalesced_group>              : public _CG_STL_NAMESPACE::true_type {};

    template <typename TyGroup>
    using scan_group_supported = _scan_group_supported<details::remove_qual<TyGroup>>;

    template <bool IsIntegralPlus>
    struct integral_optimized_scan;

    enum class ScanType { exclusive, inclusive };

    template <unsigned int GroupId,  ScanType TyScan>
    struct scan_dispatch;

    template <ScanType TyScan>
    struct scan_dispatch<details::coalesced_group_id, TyScan> {
        template <typename TyGroup, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            auto scan_result = coalesced_inclusive_scan(group, val, op);
            if (TyScan == ScanType::exclusive) {
                scan_result = convert_inclusive_to_exclusive(group,
                                                             scan_result,
                                                             _CG_STL_NAMESPACE::forward<TyVal>(val),
                                                             _CG_STL_NAMESPACE::forward<TyFn>(op));
            }
            return scan_result;
        }
    };

#if defined(_CG_CPP11_FEATURES)
    template <ScanType TyScan>
    struct scan_dispatch<details::multi_tile_group_id, TyScan> {
        template <unsigned int Size, typename ParentT, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(const thread_block_tile<Size, ParentT>& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            using warpType = details::internal_thread_block_tile<32, __static_size_multi_warp_tile_base<Size>>;
            using TyRet = details::remove_qual<TyVal>;
            const unsigned int num_warps = Size / 32;
            // In warp scan result, calculated in warp_lambda
            TyRet warp_scan;

            // In warp scan, put sum in the warp_scratch_location
            auto warp_lambda = [&] (const warpType& warp, TyRet* warp_scratch_location) {
                warp_scan = 
                    details::coalesced_inclusive_scan(warp, _CG_STL_NAMESPACE::forward<TyVal>(val), op);
                if (warp.thread_rank() + 1 == warp.size()) {
                    *warp_scratch_location = warp_scan;
                }
                if (TyScan == ScanType::exclusive) {
                    warp_scan = warp.shfl_up(warp_scan, 1);
                }
            };

            // Tile of size num_warps performing the final scan part (exclusive scan of warp sums), other threads will add it
            // to its in-warp scan result
            auto inter_warp_lambda =
                [&] (const details::internal_thread_block_tile<num_warps, warpType>& subwarp, TyRet* thread_scratch_location) {
                    auto thread_val = *thread_scratch_location;
                    auto result = coalesced_inclusive_scan(subwarp, thread_val, op);
                    *thread_scratch_location = convert_inclusive_to_exclusive(subwarp, result, thread_val, op);
            };

            TyRet previous_warps_sum = details::multi_warp_collectives_helper<TyRet>(group, warp_lambda, inter_warp_lambda);
            if (TyScan == ScanType::exclusive && warpType::thread_rank() == 0) {
                return previous_warps_sum;
            }
            if (warpType::meta_group_rank() == 0) {
                return warp_scan;
            }
            else {
                return op(warp_scan, previous_warps_sum);
            }
        }
    };

#if defined(_CG_HAS_STL_ATOMICS)
    template <unsigned int GroupId,  ScanType TyScan>
    struct scan_update_dispatch;

    template <ScanType TyScan>
    struct scan_update_dispatch<details::coalesced_group_id, TyScan> {
        template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(const TyGroup& group, TyAtomic& dst, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            details::remove_qual<TyVal> old;

            // Do regular in group scan
            auto scan_result = details::coalesced_inclusive_scan(group, val, op);

            // Last thread updates the atomic and distributes its old value to other threads
            if (group.thread_rank() == group.size() - 1) {                                                
                old = atomic_update(dst, scan_result, _CG_STL_NAMESPACE::forward<TyFn>(op));
            }
            old = group.shfl(old, group.size() - 1);
            if (TyScan == ScanType::exclusive) {
                scan_result = convert_inclusive_to_exclusive(group, scan_result, _CG_STL_NAMESPACE::forward<TyVal>(val), op);
            }
            scan_result = op(old, scan_result);
            return scan_result;
        }
    };

    template <ScanType TyScan>
    struct scan_update_dispatch<details::multi_tile_group_id, TyScan> {
        template <unsigned int Size, typename ParentT, typename TyAtomic, typename TyVal, typename TyFn>
        _CG_STATIC_QUALIFIER auto scan(const thread_block_tile<Size, ParentT>& group, TyAtomic& dst, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
            using warpType = details::internal_thread_block_tile<32, __static_size_multi_warp_tile_base<Size>>;
            using TyRet = details::remove_qual<TyVal>;
            const unsigned int num_warps = Size / 32;
            // In warp scan result, calculated in warp_lambda
            TyRet warp_scan;

            // In warp scan, put sum in the warp_scratch_location
            auto warp_lambda = [&] (const warpType& warp, TyRet* warp_scratch_location) {
                warp_scan = 
                    details::coalesced_inclusive_scan(warp, _CG_STL_NAMESPACE::forward<TyVal>(val), op);
                if (warp.thread_rank() + 1 == warp.size()) {
                    *warp_scratch_location = warp_scan;
                }
                if (TyScan == ScanType::exclusive) {
                    warp_scan = warp.shfl_up(warp_scan, 1);
                }
            };

            // Tile of size num_warps performing the final scan part (exclusive scan of warp sums), other threads will add it
            // to its in-warp scan result
            auto inter_warp_lambda =
                [&] (const details::internal_thread_block_tile<num_warps, warpType>& subwarp, TyRet* thread_scratch_location) {
                    auto thread_val = *thread_scratch_location;
                    auto scan_result = details::coalesced_inclusive_scan(subwarp, thread_val, op);
                    TyRet offset;
                    // Single thread does the atomic update with sum of all contributions and reads the old value.
                    if (subwarp.thread_rank() == subwarp.size() - 1) {
                        offset = details::atomic_update(dst, scan_result, op);
                    }
                    offset = subwarp.shfl(offset, subwarp.size() - 1);
                    scan_result = convert_inclusive_to_exclusive(subwarp, scan_result, thread_val, op);
                    // Add offset read from the atomic to the scanned warp sum.
                    // Skipping first thread, since it got defautly constructed value from the conversion,
                    // it should just return the offset received from the thread that did the atomic update.
                    if (subwarp.thread_rank() != 0) {
                        offset = op(scan_result, offset);
                    }
                    *thread_scratch_location = offset;
            };

            TyRet previous_warps_sum = details::multi_warp_collectives_helper<TyRet>(group, warp_lambda, inter_warp_lambda);
            if (TyScan == ScanType::exclusive && warpType::thread_rank() == 0) {
                return previous_warps_sum;
            }
            return op(warp_scan, previous_warps_sum);
        }
    };
#endif
#endif

    template <typename TyGroup, typename TyInputVal, typename TyRetVal>
    _CG_QUALIFIER void check_scan_params() {
        static_assert(details::is_op_type_same<TyInputVal, TyRetVal>::value, "Operator input and output types differ");
        static_assert(details::scan_group_supported<TyGroup>::value, "This group does not exclusively represent a tile");
    }

#if defined(_CG_HAS_STL_ATOMICS)
    template <typename TyGroup, typename TyDstVal, typename TyInputVal, typename TyRetVal>
    _CG_QUALIFIER void check_scan_update_params() {
        check_scan_params<TyGroup, TyInputVal, TyRetVal>();
        static_assert(details::is_op_type_same<TyDstVal, TyInputVal>::value, "Destination and input types differ");
    }
#endif

} // details

template <typename TyGroup, typename TyVal, typename TyFn>
_CG_QUALIFIER auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_scan_params<TyGroup, TyVal, decltype(op(val, val))>();

    using dispatch = details::scan_dispatch<TyGroup::_group_id, details::ScanType::inclusive>;
    return dispatch::scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template <typename TyGroup, typename TyVal>
_CG_QUALIFIER details::remove_qual<TyVal> inclusive_scan(const TyGroup& group, TyVal&& val) {
    return inclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), cooperative_groups::plus<details::remove_qual<TyVal>>());
}

template <typename TyGroup, typename TyVal, typename TyFn>
_CG_QUALIFIER auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_scan_params<TyGroup, TyVal, decltype(op(val, val))>();

    using dispatch = details::scan_dispatch<TyGroup::_group_id, details::ScanType::exclusive>;
    return dispatch::scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template <typename TyGroup, typename TyVal>
_CG_QUALIFIER details::remove_qual<TyVal> exclusive_scan(const TyGroup& group, TyVal&& val) {
    return exclusive_scan(group, _CG_STL_NAMESPACE::forward<TyVal>(val), cooperative_groups::plus<details::remove_qual<TyVal>>());
}

#if defined(_CG_HAS_STL_ATOMICS)
template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco, typename TyFn>
_CG_QUALIFIER auto inclusive_scan_update(const TyGroup& group, cuda::atomic<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_scan_update_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();

    using dispatch = details::scan_update_dispatch<TyGroup::_group_id, details::ScanType::inclusive>;
    return dispatch::scan(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco>
_CG_QUALIFIER TyVal inclusive_scan_update(const TyGroup& group, cuda::atomic<TyVal, Sco> & dst, TyInputVal&& val) {
    return inclusive_scan_update(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), cooperative_groups::plus<TyVal>());
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco, typename TyFn>
_CG_QUALIFIER auto exclusive_scan_update(const TyGroup& group, cuda::atomic<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_scan_update_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();

    using dispatch = details::scan_update_dispatch<TyGroup::_group_id, details::ScanType::exclusive>;
    return dispatch::scan(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco>
_CG_QUALIFIER TyVal exclusive_scan_update(const TyGroup& group, cuda::atomic<TyVal, Sco>& dst, TyInputVal&& val) {
    return exclusive_scan_update(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), cooperative_groups::plus<TyVal>());
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco, typename TyFn>
_CG_QUALIFIER auto inclusive_scan_update(const TyGroup& group, const cuda::atomic_ref<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_scan_update_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();

    using dispatch = details::scan_update_dispatch<TyGroup::_group_id, details::ScanType::inclusive>;
    return dispatch::scan(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco>
_CG_QUALIFIER TyVal inclusive_scan_update(const TyGroup& group, const cuda::atomic_ref<TyVal, Sco> & dst, TyInputVal&& val) {
    return inclusive_scan_update(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), cooperative_groups::plus<TyVal>());
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco, typename TyFn>
_CG_QUALIFIER auto exclusive_scan_update(const TyGroup& group, const cuda::atomic_ref<TyVal, Sco>& dst, TyInputVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    details::check_scan_update_params<TyGroup, TyVal, details::remove_qual<TyInputVal>, decltype(op(val, val))>();

    using dispatch = details::scan_update_dispatch<TyGroup::_group_id, details::ScanType::exclusive>;
    return dispatch::scan(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), _CG_STL_NAMESPACE::forward<TyFn>(op));
}

template<typename TyGroup, typename TyVal, typename TyInputVal, cuda::thread_scope Sco>
_CG_QUALIFIER TyVal exclusive_scan_update(const TyGroup& group, const cuda::atomic_ref<TyVal, Sco>& dst, TyInputVal&& val) {
    return exclusive_scan_update(group, dst, _CG_STL_NAMESPACE::forward<TyInputVal>(val), cooperative_groups::plus<TyVal>());
}
#endif

_CG_END_NAMESPACE

#endif // _CG_SCAN_H_
