from os import system
# g++ -std=c++17 -Ofast -c extension.cpp -o ./output/main.o -pthread -fPIC
def s(x):
    print(x)
    system(x)
files = ["BatchMCTS.cpp", "Constants.cpp", "MCTS.cpp", "position.cpp", "tables.cpp", "tests.cpp", "types.cpp", "extension_BatchMCTS.cpp"]
files = [f.replace(".cpp", "") for f in files]
for f in files:
    s("g++ -std=c++17 -Ofast -fPIC -c {0} -o ./output/{1}".format(f + ".cpp", f + ".o"))
files = ["./output/" + f + ".o" for f in files]
s("g++ -shared -o ./output/extension_BatchMCTS.so {0}".format(" ".join(files)))