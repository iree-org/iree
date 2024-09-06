import sys
import re

key = str(sys.argv[1])

f_ = open("build_tools/ccache/fuse_connection.yaml")
s = f_.read()
f_.close()

s = re.sub("CCACHE_KEY", key, s)
f_ = open("build_tools/ccache/fuse_connection2.yaml", "w+")
f_.write(s)
f_.close()
