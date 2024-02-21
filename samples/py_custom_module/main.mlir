module @main {
  util.global private @tokens_list : !util.list<?>
  func.func private @detokenizer.accumtokens(tensor<?xi32>, !util.list<?>) -> i32
  func.func private @detokenizer.jointokens(!util.list<?>) -> !util.buffer
  func.func private @detokenizer.reset()

  util.initializer {
    %capacity = arith.constant 25 : index
    %0 = util.list.create %capacity : !util.list<?>
    util.global.store %0, @tokens_list : !util.list<?>
    util.return
  }

  func.func public @add_tokens(%ids : tensor<?xi32>) -> i32 {
    %lst = util.global.load @tokens_list : !util.list<?>
    %count = call @detokenizer.accumtokens(%ids, %lst) : (tensor<?xi32>, !util.list<?>) -> (i32)
    return %count : i32
  }

  func.func public @reset() {
    %zero = arith.constant 0 : index
    %lst = util.global.load @tokens_list : !util.list<?>
    util.list.resize %lst, %zero : !util.list<?>
    call @detokenizer.reset() : () -> ()
    return
  }

  func.func public @get_results() -> !util.buffer {
    %lst = util.global.load @tokens_list : !util.list<?>
    %buffer = call @detokenizer.jointokens(%lst) : (!util.list<?>) -> (!util.buffer)
    return %buffer : !util.buffer
  }
}
