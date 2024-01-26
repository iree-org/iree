// RUN: iree-opt --split-input-file --iree-vm-conversion --iree-vm-target-index-bits=32 --allow-unregistered-dialect %s | FileCheck %s

//      CHECK: vm.initializer {
// CHECK-NEXT:  vm.return
// CHECK-NEXT: }
util.initializer {
  util.return
}

// -----

// CHECK: vm.func private @fn(%[[ARG0:.+]]: i32) -> i32
util.func private @fn(%arg0: i32) -> (i32) {
  // CHECK: vm.return %[[ARG0]] : i32
  util.return %arg0 : i32
}

// -----

// CHECK: vm.func private @fn_noinline(%[[ARG0:.+]]: i32) -> i32
// CHECK-SAME: attributes {inlining_policy = #util.inline.never}
util.func private @fn_noinline(%arg0: i32) -> i32 attributes {
  inlining_policy = #util.inline.never
} {
  // CHECK: vm.return %[[ARG0]] : i32
  util.return %arg0 : i32
}

// -----

// CHECK: vm.func private @fn_reflection
// CHECK-SAME: iree.reflection = {f = "FOOBAR"}
util.func @fn_reflection(%arg0: i32) -> i32 attributes {
  iree.reflection = {f = "FOOBAR"}
} {
  util.return %arg0 : i32
}

// -----

// CHECK: vm.func private @internal_function_name
// CHECK: vm.export @internal_function_name
util.func @internal_function_name(%arg0: i32) -> i32 {
  util.return %arg0 : i32
}

// -----

// CHECK: vm.import private @some.import(i32) -> !vm.buffer
// CHECK-SAME: attributes {minimum_version = 4 : i32}
util.func private @some.import(%arg0: index) -> !util.buffer attributes {
  vm.version = 4 : i32
}
// CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> !vm.buffer
util.func @my_fn(%arg0: index) -> !util.buffer {
  // CHECK: %[[RET:.+]] = vm.call @some.import(%[[ARG0]]) : (i32) -> !vm.buffer
  %0 = util.call @some.import(%arg0) : (index) -> !util.buffer
  // CHECK: vm.return %[[RET]]
  util.return %0 : !util.buffer
}

// -----

// CHECK: vm.import private @signatured.import(i64) -> i64
util.func private @signatured.import(%arg0: index) -> index attributes {
  vm.signature = (i64) -> i64
}
// CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
util.func @my_fn(%arg0: index) -> index {
  // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
  // CHECK: %[[IMPORT_RET:.+]] = vm.call @signatured.import(%[[IMPORT_ARG]]) : (i64) -> i64
  %0 = util.call @signatured.import(%arg0) : (index) -> index
  // CHECK: %[[RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
  // CHECK: vm.return %[[RET]] : i32
  util.return %0 : index
}

// -----

util.func private @other.fn_i32(%arg0: i32) -> i32
// CHECK: vm.func private @my_fn
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
util.func @my_fn(%arg0 : i32) -> i32 {
  // CHECK: vm.call @other.fn_i32(%[[ARG0]]) : (i32) -> i32
  %0 = util.call @other.fn_i32(%arg0) : (i32) -> i32
  util.return %0 : i32
}

// -----

util.func private @other.fn_i1(%arg0: i1) -> i1
// CHECK: vm.func private @my_fn
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
util.func @my_fn(%arg0 : i1) -> i1 {
  // CHECK: vm.call @other.fn_i1(%[[ARG0]]) : (i32) -> i32
  %0 = util.call @other.fn_i1(%arg0) : (i1) -> i1
  util.return %0 : i1
}

// -----

// NOTE: we require conversion for the import but not the fallback!
// CHECK: vm.import private optional @optional.import(i64) -> i64
util.func private @optional.import(%arg0: index) -> index attributes {
  vm.signature = (i64) -> i64,
  vm.fallback = @some_fallback
}
// CHECK: vm.func private @some_fallback(%{{.+}}: i32) -> i32
util.func private @some_fallback(%arg0: index) -> index {
  util.return %arg0 : index
}
// CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
util.func @my_fn(%arg0: index) -> index {
  // CHECK: %[[RESOLVED:.+]] = vm.import.resolved @optional.import
  // CHECK: vm.cond_br %[[RESOLVED]], ^bb1, ^bb2
  %0 = util.call @optional.import(%arg0) : (index) -> index

  // CHECK: ^bb1:
  // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
  // CHECK: %[[IMPORT_RET:.+]] = vm.call @optional.import(%[[IMPORT_ARG]]) : (i64) -> i64
  // CHECK: %[[BB1_RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
  // CHECK: vm.br ^bb3(%[[BB1_RET]] : i32)

  // CHECK: ^bb2:
  // CHECK: %[[FALLBACK_RET:.+]] = vm.call @some_fallback(%[[ARG0]]) : (i32) -> i32
  // CHECK: vm.br ^bb3(%[[FALLBACK_RET]] : i32)

  // CHECK: ^bb3(%[[RET:.+]]: i32):
  // CHECK: vm.return %[[RET]]
  util.return %0 : index
}

// -----

// NOTE: we require conversion for the import but not the fallback!
// CHECK: vm.import private optional @optional.import(i64) -> i64
util.func private @optional.import(%arg0: index) -> index attributes {
  vm.signature = (i64) -> i64,
  vm.fallback = @fallback.import
}
// CHECK: vm.import private @fallback.import(i64) -> i64
util.func private @fallback.import(%arg0: index) -> index attributes {
  vm.signature = (i64) -> i64
}
// CHECK: vm.func private @my_fn(%[[ARG0:.+]]: i32) -> i32
util.func @my_fn(%arg0: index) -> index {
  // CHECK: %[[RESOLVED:.+]] = vm.import.resolved @optional.import
  // CHECK: vm.cond_br %[[RESOLVED]], ^bb1, ^bb2
  %0 = util.call @optional.import(%arg0) : (index) -> index

  // CHECK: ^bb1:
  // CHECK: %[[IMPORT_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
  // CHECK: %[[IMPORT_RET:.+]] = vm.call @optional.import(%[[IMPORT_ARG]]) : (i64) -> i64
  // CHECK: %[[BB1_RET:.+]] = vm.trunc.i64.i32 %[[IMPORT_RET]]
  // CHECK: vm.br ^bb3(%[[BB1_RET]] : i32)

  // CHECK: ^bb2:
  // CHECK: %[[FALLBACK_ARG:.+]] = vm.ext.i32.i64.s %[[ARG0]]
  // CHECK: %[[FALLBACK_RET:.+]] = vm.call @fallback.import(%[[FALLBACK_ARG]]) : (i64) -> i64
  // CHECK: %[[BB2_RET:.+]] = vm.trunc.i64.i32 %[[FALLBACK_RET]]
  // CHECK: vm.br ^bb3(%[[BB2_RET]] : i32)

  // CHECK: ^bb3(%[[RET:.+]]: i32):
  // CHECK: vm.return %[[RET]]
  util.return %0 : index
}

// -----

// CHECK: vm.import private @opaque.import() -> !vm.ref<!some.type<foo>>
util.func private @opaque.import() -> !some.type<foo>
// CHECK: vm.func private @my_fn() -> !vm.ref<!some.type<foo>>
util.func @my_fn() -> !some.type<foo> {
  // CHECK: %[[RET:.+]] = vm.call @opaque.import() : () -> !vm.ref<!some.type<foo>>
  %0 = util.call @opaque.import() : () -> !some.type<foo>
  // CHECK: vm.return %[[RET]]
  util.return %0 : !some.type<foo>
}
