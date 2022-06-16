set(MINILM_I32_MODULE
  NAME
    "MiniLM"
  TAGS
    "i32"
  SOURCE
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased.
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-tf-model.tar.gz"
  ENTRY_FUNCTION
    "predict"
  FUNCTION_INPUTS
    "1x512xi32,1x512xi32,1x512xi32"
)

include(linux-x86_64.cmake)
