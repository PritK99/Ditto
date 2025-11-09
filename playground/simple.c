// This is a rigorous test of preprocessing code 
#include <stdio.h>
#include <string.h>

// Global variable
int x = 100;

// Macros
#define MUL(a, b) ((a) * (b))
#define PRINT_INT(x) printf("%d\n", (x))

// Struct definition
struct Car {
    char* number_plate;
    char* model_name;
    int price;
    int year_of_manufacture;
};

// Function to square a number
int square(int x) {
    printf("Squaring the number: %d\n", 5);
    return MUL(x, x);
}

// Function to print Car info
void print_car(struct Car c) {
    printf("Car: %s, Model: %s, Price: %d, Year: %d\n",
           c.number_plate, c.model_name, c.price, c.year_of_manufacture);
}

int main() {
    // Example of using the square function
    int r = square(5);
    PRINT_INT(r);

    // Example of using the struct
    struct Car car1;
    car1.number_plate = "ABC123";
    car1.model_name = "Tesla Model S";
    car1.price = 80000;
    car1.year_of_manufacture = 2022;

    print_car(car1);

    return 0;
}
