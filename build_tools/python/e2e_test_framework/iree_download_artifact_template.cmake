# Fetch the model from "$__SOURCE_URL"
iree_fetch_artifact(
  NAME
    "$__TARGET_NAME"
  SOURCE_URL
    "$__SOURCE_URL"
  OUTPUT
    "$__OUTPUT_PATH"
  UNPACK
)
