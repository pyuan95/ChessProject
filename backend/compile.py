from os import system
import sys

# g++ -std=c++17 -Ofast -c extension.cpp -o ./output/main.o -pthread -fPIC
def s(x):
    print(x)
    system(x)


if len(sys.argv) >= 2:
    opt = ""
else:
    opt = "-Ofast"

files = [
    "BatchMCTS.cpp",
    "Constants.cpp",
    "MCTS.cpp",
    "position.cpp",
    "tables.cpp",
    "tests.cpp",
    "types.cpp",
    "extension_BatchMCTS.cpp",
    "tbprobe.cpp",
    "tablebase_evaluation.cpp",
    "memmanager.cpp",
]
files = [f.replace(".cpp", "") for f in files]
for f in files:
    s("g++ -std=c++17 {0} -fPIC -c {1} -o ./output/{2} -pthread".format(opt, f + ".cpp", f + ".o"))
files = ["./output/" + f + ".o" for f in files]
s("g++ -shared -o ./output/extension_BatchMCTS.so {0}".format(" ".join(files)))
