#!/usr/bin/env python3

# Usage:
#   python array.py <shape_before_transposition> <transposition> <dtype>
# Where:
#   <shape_before_transposition> is a comma-separated list of ints.
#   <transposition> is a comma-separated list of ints.
#   <dtype> is the MLIR element type, e.g. f32
# Example:
#   python array.py 4,8 1,0 int

import sys
import numpy as np
import re

if len(sys.argv) != 4:
  print("Expected 3 command line args. See file comment.")
  sys.exit(1)

shape_before_transposition = [int(x) for x in sys.argv[1].split(',')]
transposition = [int(x) for x in sys.argv[2].split(',')]
dtype = float if 'f' in sys.argv[3] else int

if len(shape_before_transposition) != len(transposition):
  print("shape and transposition should have the same length")
  sys.exit(1)

flatsize = np.prod(np.array(shape_before_transposition))
modulus = 256 if sys.argv[3] == 'i8' else 65536
lin=np.linspace(0, flatsize - 1, flatsize, dtype=dtype)
mod=np.mod(lin, modulus)
expanded=np.reshape(mod, shape_before_transposition)
transposed=np.transpose(expanded, transposition)
np.set_printoptions(threshold = sys.maxsize, suppress=True)
repr=np.array_repr(transposed, max_line_width=1024)

print(f'      // array.py {' '.join(sys.argv[1:4])}')
print(repr.replace('array(', '      ').replace(')', ' '))
