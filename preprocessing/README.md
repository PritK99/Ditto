1) Issue with functions being labelled as variables
2) Issues with standard library functions being obfuscated 
3) Dealing with statements like `printf("\033[38;2;%06X;%d;%d;%dm ", color, x, y, 0)`. The tokenizer here converts "%d %d %d" as var1 which is incorrect as this is part of standard library requirements and not a variable.
4) In C++, using namespace std makes std as var0. It should appear as std.
5) In C++, int main should appear as it is and not int var1