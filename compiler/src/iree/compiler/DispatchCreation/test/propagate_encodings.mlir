// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-encodings))" --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_encoding_through_tensor_cast(%src: tensor<1024x?xf16>) -> tensor<?x512xf16, #encoding> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %0 = iree_encoding.set_encoding %cast : tensor<?x512xf16> -> tensor<?x512xf16, #encoding>
  util.return %0 : tensor<?x512xf16, #encoding>
}

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_encoding_through_tensor_cast(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]

// -----

#encoding = #iree_encoding.layout<[#iree_encoding.testing<>]>
util.func @dont_propagate_unserialized_layout(%src: tensor<1024x?xf16>) -> tensor<?x512xf16, #encoding> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %0 = iree_encoding.set_encoding %cast : tensor<?x512xf16> -> tensor<?x512xf16, #encoding>
  util.return %0 : tensor<?x512xf16, #encoding>
}

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @dont_propagate_unserialized_layout(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SRC]] : tensor<1024x?xf16> to tensor<?x512xf16>
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[CAST]] : tensor<?x512xf16> -> tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[SET_ENCODING]]

// -----

// Test that propagation works with encoding_dims when they dominate the insertion point.
// The encoding_dims are function arguments, so they always dominate.

#encoding_with_dims = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_encoding_with_encoding_dims(%src: tensor<1024x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x512xf16, #encoding_with_dims> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %0 = iree_encoding.set_encoding %cast encoding_dims{%m, %n, %k} : tensor<?x512xf16> -> tensor<?x512xf16, #encoding_with_dims>
  util.return %0 : tensor<?x512xf16, #encoding_with_dims>
}

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_encoding_with_encoding_dims(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<1024x?xf16>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[K:[a-zA-Z0-9]+]]: index
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] encoding_dims{%[[M]], %[[N]], %[[K]]} : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]

// -----

// Test that propagation works with encoding_dims that need rematerialization.
// Here, the encoding_dims are computed from the cast result, but can be
// rematerialized as tensor.dim on the cast source (which then folds to constants
// for static dimensions).

#encoding_remat = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_with_rematerialized_encoding_dims(%src: tensor<1024x?xf16>) -> tensor<?x512xf16, #encoding_remat> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %c0 = arith.constant 0 : index
  // This dim is computed from %cast, but can be rematerialized from %src
  %m = tensor.dim %cast, %c0 : tensor<?x512xf16>
  %c512 = arith.constant 512 : index
  %0 = iree_encoding.set_encoding %cast encoding_dims{%m, %c512} : tensor<?x512xf16> -> tensor<?x512xf16, #encoding_remat>
  util.return %0 : tensor<?x512xf16, #encoding_remat>
}

// The tensor.dim is rematerialized as tensor.dim on %src, which folds to constant 1024.
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_with_rematerialized_encoding_dims(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:     %[[C512:.+]] = arith.constant 512 : index
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] encoding_dims{%[[C1024]], %[[C512]]} : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]

// -----

// Test that propagation works when encoding_dims use tensor.dim on the
// propagation source. The dim is rematerialized from the new source tensor.

#encoding_dim_on_source = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_with_dim_on_propagation_source(
    %src: tensor<1024x?xf16>) -> tensor<?x512xf16, #encoding_dim_on_source> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %c0 = arith.constant 0 : index
  // This dim depends on the cast result (propagation source).
  // It gets rematerialized to use the new source (src).
  %m = tensor.dim %cast, %c0 : tensor<?x512xf16>
  %c512 = arith.constant 512 : index
  %0 = iree_encoding.set_encoding %cast encoding_dims{%m, %c512} : tensor<?x512xf16> -> tensor<?x512xf16, #encoding_dim_on_source>
  util.return %0 : tensor<?x512xf16, #encoding_dim_on_source>
}

// Propagation succeeds because tensor.dim on propagation source (cast) is
// rematerialized to use the new source (src). Since dim 0 is statically known
// (1024), a constant is created instead of tensor.dim.
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_with_dim_on_propagation_source(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<1024x?xf16>
// CHECK-DAG:     %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:     %[[C512:.+]] = arith.constant 512 : index
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] encoding_dims{%[[C1024]], %[[C512]]} : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]]
// CHECK:         util.return %[[CAST]]

// -----

// Test that propagation works when encoding_dims are arithmetic ops with
// dominating operands. The arith.addi can be cloned since its operands
// (function arguments) dominate the new insertion point.

#encoding_arith = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_with_arith_encoding_dims(
    %src: tensor<1024x?xf16>, %m: index, %n: index) -> tensor<?x512xf16, #encoding_arith> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  // This arith.addi can be cloned since %m and %n dominate the cast
  %sum = arith.addi %m, %n : index
  %0 = iree_encoding.set_encoding %cast encoding_dims{%sum} : tensor<?x512xf16> -> tensor<?x512xf16, #encoding_arith>
  util.return %0 : tensor<?x512xf16, #encoding_arith>
}

// The arith.addi is cloned at the new insertion point.
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_with_arith_encoding_dims(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<1024x?xf16>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK:         %[[SUM:.+]] = arith.addi %[[M]], %[[N]] : index
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] encoding_dims{%[[SUM]]} : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]

// -----

// Test that propagation works when encoding_dims use tensor.dim on a tensor
// that already dominates (not the propagation source). The tensor.dim on %src
// already dominates the cast, so no rematerialization is needed.

#encoding_dim_dominates = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_with_dim_on_dominating_tensor(
    %src: tensor<1024x?xf16>) -> tensor<?x512xf16, #encoding_dim_dominates> {
  %c1 = arith.constant 1 : index
  // tensor.dim on %src - this dominates the cast already
  %dynamic_dim = tensor.dim %src, %c1 : tensor<1024x?xf16>
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %0 = iree_encoding.set_encoding %cast encoding_dims{%dynamic_dim} : tensor<?x512xf16> -> tensor<?x512xf16, #encoding_dim_dominates>
  util.return %0 : tensor<?x512xf16, #encoding_dim_dominates>
}

// The tensor.dim on %src already dominates, so it's used directly.
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_with_dim_on_dominating_tensor(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<1024x?xf16>
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[DIM:.+]] = tensor.dim %[[SRC]], %[[C1]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] encoding_dims{%[[DIM]]} : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]

// -----

// Test that recursive rematerialization works for arith ops with operands
// that can be recursively rematerialized. The arith.muli uses %dim which is
// computed from %cast. Since %dim is a tensor.dim on the propagation source,
// it gets rematerialized from %src, and then arith.muli is cloned.

#encoding_arith_recursive = #iree_encoding.layout<[#iree_encoding.testing<layouts = []>]>
util.func @propagate_with_recursive_rematerialization(
    %src: tensor<1024x?xf16>, %multiplier: index) -> tensor<?x512xf16, #encoding_arith_recursive> {
  %cast = tensor.cast %src : tensor<1024x?xf16> to tensor<?x512xf16>
  %c0 = arith.constant 0 : index
  // %dim depends on %cast
  %dim = tensor.dim %cast, %c0 : tensor<?x512xf16>
  // arith.muli can be cloned after recursively rematerializing %dim from %src.
  %product = arith.muli %dim, %multiplier : index
  %0 = iree_encoding.set_encoding %cast encoding_dims{%product} : tensor<?x512xf16> -> tensor<?x512xf16, #encoding_arith_recursive>
  util.return %0 : tensor<?x512xf16, #encoding_arith_recursive>
}

// Propagation succeeds with recursive rematerialization of tensor.dim and arith.muli.
// Since dim 0 of src (1024) is static, a constant is created instead of tensor.dim.
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.layout{{.*}}
// CHECK-LABEL: @propagate_with_recursive_rematerialization(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<1024x?xf16>
// CHECK-SAME:    %[[MULT:[a-zA-Z0-9]+]]: index
// CHECK:         %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:         %[[PRODUCT:.+]] = arith.muli %[[MULT]], %[[C1024]] : index
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] encoding_dims{%[[PRODUCT]]} : tensor<1024x?xf16> -> tensor<1024x?xf16, #[[$ENCODING]]>
// CHECK:         %[[CAST:.+]] = tensor.cast %[[SET_ENCODING]] : tensor<1024x?xf16, #[[$ENCODING]]> to tensor<?x512xf16, #[[$ENCODING]]>
// CHECK:         util.return %[[CAST]]
