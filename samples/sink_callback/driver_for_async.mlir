module @driver {
  util.func private @foo.foo(%arg0: tensor<?x18xf32>) -> tensor<?x18xf32>
  util.func public @main(%arg0: tensor<?x18xf32>) {
    %0 = util.call @foo.foo(%arg0) : (tensor<?x18xf32>) -> tensor<?x18xf32>
    util.return
  }
}
