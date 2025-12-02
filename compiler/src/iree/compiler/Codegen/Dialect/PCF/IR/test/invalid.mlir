// RUN: iree-opt --split-input-file %s --verify-diagnostics

util.func private @scope_mismatch(%dim: index) {
// expected-error@+1 {{expected region ref argument to be of type !pcf.sref with scope #pcf.sequential}}
  pcf.generic scope(#pcf.sequential)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

// expected-note@+1 {{prior use here}}
util.func private @init_type_mismatch(%0: tensor<3xi32>) {
  pcf.generic scope(#pcf.test_scope)
// expected-error@+1 {{expects different type than prior uses: 'tensor<?xi32>' vs 'tensor<3xi32>'}}
    execute(%ref = %0)[%num_threads: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @sync_scope_mismatch(%dim: index) {
// expected-error@+1 {{expected region ref argument to have none or parent sync scope}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope, i32>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_shape_mismatch(%dim: index) {
// expected-error@+1 {{region arg at index 0 with type '!pcf.sref<3xi32, #pcf.test_scope>' shape mismatch with tied result of type 'tensor<?xi32>'}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<3xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @arg_eltype_mismatch(%dim: index) {
// expected-error@+1 {{region arg at index 0 element type mismatch of 'f32' vs 'i32'}}
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?xf32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @empty_count(%dim: index) {
// expected-error@+1 {{expected at least one iteration count argument}}
  pcf.loop scope(#pcf.sequential) count()
    execute(%ref)[]
         : (!pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>{%dim}) {
    pcf.return
  }
  util.return
}

// -----

util.func private @missing_execute_keyword() {
  pcf.generic scope(#pcf.test_scope)
    // expected-error@+1 {{custom op 'pcf.generic' expected 'execute'}}
    notexecute(%ref)[%id: index, %n: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>)
        -> (tensor<4xi32>) {
    pcf.return
  }
  util.return
}

// -----

util.func private @result_not_shaped_type() {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>)
        // expected-error@+1 {{custom op 'pcf.generic' result type must be a shaped type}}
        -> (i32) {
    pcf.return
  }
  util.return
}

// -----

util.func private @dynamic_dim_mismatch(%dim0: index) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.test_scope>)
        // expected-error@+1 {{custom op 'pcf.generic' expected 2 dynamic dimension operands for type 'tensor<?x?xi32>', but got 1}}
        -> (tensor<?x?xi32>{%dim0}) {
    pcf.return
  }
  util.return
}
