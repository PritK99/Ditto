#include <stdio.h>

int x = 100;
#define MUL(a, b) ((a) * (b))
#define PRINT_INT(x) printf("%d\n", (x))

int square(int x) {
    printf("%d", 5);
    return MUL(x, x);
}

int main() {
    int r = square(5);
    PRINT_INT(r);
    return 0;
}