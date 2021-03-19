import sys
import re

tag = sys.argv[1]
version_pattern = re.compile("\d+(.\d+)+")
version = re.search(version_pattern, tag).group()

with open("knodle/version.py", "w") as fp:
    fp.writelines([
        "# this is an auto-generated file on release\n",
        f"__version__ = '{version}'\n"
    ])