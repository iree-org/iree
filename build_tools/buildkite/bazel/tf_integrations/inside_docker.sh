cd integrations/tensorflow
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
BAZEL_BINDIR="$(${BAZEL_CMD[@]?} info bazel-bin)"
"${BAZEL_CMD[@]?}" query //iree_tf_compiler/... | \
   xargs "${BAZEL_CMD[@]?}" test --config=generic_clang \
      --test_tag_filters="-nokokoro" \
      --build_tag_filters="-nokokoro"
