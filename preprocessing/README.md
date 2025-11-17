# Preprocessing

This section deals with conversion of code in raw form to tokens. 

## Methodology

## Requirements

1) Code Obfuscation: All the user defined entities i.e. variables, user defined functions, and literals, should be obfuscated like `var0`, `func0`, and `lit0`.

2) However, the obfuscation process should not affect standard functions such as `printf`, `main` and other keywords. 

3) The tokenization process should take care of escape characters.

4) Special cases need to be made for dealing with entities like `#define`, `printf("%d", ...)`, etc. 

5) The tokenization process should return:

    * Tokens
    * Raw Positions
    * LCA Positions
    * Variable Map
    * Literal Map
    * Function Map

    Hence the expected output is a CSV with all these fields.

6) Because we know we are dealing with programming languages, we know our vocabulory size can not be big. Similarly, our token size can not be big. This is because there can only be a few keywords that can be used in a language, and the keywords are generally short enough.

7) The tokenizer should only return tokens for the code that is correct and executable. 

## Implementation Details

**Note 1**: `preprocess_parallel.py` for C++ code takes a lot of time. This is because some codes contain around 15L - 20L characters. Due to this, the progress bar might not update for long time. However, this THE time taken to compute tokens for the massive code files. For C, processing tokens required around 3 hours using 10 CPU cores. This was way faster compared to C++ because of two reasons. First, the C code snippets that we have collected are not as large as C++ code snippets. Second, parsing C is easier than C++. For C++, processing tokens required around 60 hours using 15 CPU cores.

**Note 2**: The following code files were discarded because these are not compilable.

C: `[
    258, 3982, 15852, 21382, 27197, 29811, 31107, 33128, 34797, 37503,
    50880, 58387, 62001, 65918, 66740, 75154, 77299, 77369, 79863, 81790,
    91034, 91224, 99925, 105763, 106437, 109104, 114266, 117244
]`

C++: `[
    401, 1333, 2260, 3169, 3456, 3950, 4711, 5689, 6119, 6222,
    7049, 8168, 10547, 12281, 13489, 14111, 14687, 16843, 17249,
    19101, 19164, 19303, 20703, 21714, 21787, 22731, 23080, 23729,
    24660, 25600, 25791, 25938, 27710, 27962, 28126, 30023, 30516,
    30569, 31150, 31665, 31763, 32167, 32354, 33767, 34818, 34989,
    36071, 38537, 38711, 39354, 40207, 43347, 44589, 48412, 48829,
    50084, 51873, 52124, 52740, 55837, 56396, 57183, 58133, 60596,
    61114, 61593, 61674
]`

## Examples

For `samples/simple.c` file, 

```c
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
```

```
 ---- Transformed Code ----
['int', 'var0', '=', 'lit0', ';', 'struct', 'struct0', '{', 'char', '*', 'var1', ';', 'char', '*', 'var2', ';', 'int', 'var3', ';', 'int', 'var4', ';', '}', ';', 'int', 'func0', '(', 'int', 'var0', ')', '{', 'printf', '(', 'lit1', ',', 'lit2', ')', ';', 'return', '(', '(', 'var0', ')', '*', '(', 'var0', ')', ')', ';', '}', 'void', 'func1', '(', 'struct', 'struct0', 'var5', ')', '{', 'printf', '(', 'lit3', ',', 'var5', '.', 'var1', ',', 'var5', '.', 'var2', ',', 'var5', '.', 'var3', ',', 'var5', '.', 'var4', ')', ';', '}', 'int', 'main', '(', ')', '{', 'int', 'var6', '=', 'func0', '(', 'lit2', ')', ';', 'printf', '(', 'lit4', ',', '(', 'var6', ')', ')', ';', 'struct', 'struct0', 'var7', ';', 'var7', '.', 'var1', '=', 'lit5', ';', 'var7', '.', 'var2', '=', 'lit6', ';', 'var7', '.', 'var3', '=', 'lit7', ';', 'var7', '.', 'var4', '=', 'lit8', ';', 'func1', '(', 'var7', ')', ';', 'return', 'lit9', ';', '}']

---- Variable Dict ----
{'x': 'var0', 'number_plate': 'var1', 'model_name': 'var2', 'price': 'var3', 'year_of_manufacture': 'var4', 'c': 'var5', 'r': 'var6', 'car1': 'var7'}

---- Function Dict ----
{'square': 'func0', 'print_car': 'func1'}

---- Literal Dict ----
{'100': 'lit0', '"Squaring the number: %d\\n"': 'lit1', '5': 'lit2', '"Car: %s, Model: %s, Price: %d, Year: %d\\n"': 'lit3', '"%d\\n"': 'lit4', '"ABC123"': 'lit5', '"Tesla Model S"': 'lit6', '80000': 'lit7', '2022': 'lit8', '0': 'lit9'}

---- Struct Dict ----
{'Car': 'struct0'}

---- Class Dict ----
{}
Final output written to cleaned_simple.c
```

For `samples/simple.cpp` file,

```cpp
// This is a rigorous test of preprocessing code 
#include <iostream>
#include <vector>
#include <string>

// Base class
class Vehicle {
public:
    std::string number_plate;
    std::string model_name;
    int price;
    int year_of_manufacture;

    Vehicle(const std::string& num, const std::string& model, int p, int year)
        : number_plate(num), model_name(model), price(p), year_of_manufacture(year) {}

    virtual void printInfo() const {
        std::cout << "Vehicle: " << number_plate
                  << ", Model: " << model_name
                  << ", Price: $" << price
                  << ", Year: " << year_of_manufacture << "\n";
    }

    virtual ~Vehicle() = default; // Always good to have virtual destructor in base class
};

// Derived class
class Car : public Vehicle {
public:
    int num_doors;

    Car(const std::string& num, const std::string& model, int p, int year, int doors)
        : Vehicle(num, model, p, year), num_doors(doors) {}

    void printInfo() const override {
        Vehicle::printInfo();
        std::cout << "Number of doors: " << num_doors << "\n";
    }
};

// Function to square a number
int square(int x) {
    std::cout << "Squaring the number: " << x << "\n";
    return x * x;
}



int main() {
    // Using square function
    int r = square(5);
    std::cout << "Result: " << r << "\n";

    // Using vectors and inheritance
    std::vector<Vehicle*> vehicles;

    // Add a Car object
    vehicles.push_back(new Car("ABC123", "Tesla Model S", 80000, 2022, 4));
    vehicles.push_back(new Vehicle("XYZ789", "Generic Vehicle", 30000, 2015));

    // Print info for all vehicles
    for (const auto& v : vehicles) {
        v->printInfo();
        std::cout << "--------------------\n";
    }

    // Clean up
    for (auto v : vehicles) delete v;

    return 0;
}
```

```
---- Transformed Code ----
['class', 'class0', '{', 'public', ':', 'std', '::', 'string', 'var0', ';', 'std', '::', 'string', 'var1', ';', 'int', 'var2', ';', 'int', 'var3', ';', 'class0', '(', 'const', 'std', '::', 'string', '&', 'var4', ',', 'const', 'std', '::', 'string', '&', 'var5', ',', 'int', 'var6', ',', 'int', 'var7', ')', ':', 'var0', '(', 'var4', ')', ',', 'var1', '(', 'var5', ')', ',', 'var2', '(', 'var6', ')', ',', 'var3', '(', 'var7', ')', '{', '}', 'virtual', 'void', 'func0', '(', ')', 'const', '{', 'std', '::', 'cout', '<<', 'lit0', '<<', 'var0', '<<', 'lit1', '<<', 'var1', '<<', 'lit2', '<<', 'var2', '<<', 'lit3', '<<', 'var3', '<<', 'lit4', ';', '}', 'virtual', '~', 'class0', '(', ')', '=', 'default', ';', '}', ';', 'class', 'class1', ':', 'public', 'class0', '{', 'public', ':', 'int', 'var8', ';', 'class1', '(', 'const', 'std', '::', 'string', '&', 'var4', ',', 'const', 'std', '::', 'string', '&', 'var5', ',', 'int', 'var6', ',', 'int', 'var7', ',', 'int', 'var9', ')', ':', 'class0', '(', 'var4', ',', 'var5', ',', 'var6', ',', 'var7', ')', ',', 'var8', '(', 'var9', ')', '{', '}', 'void', 'func0', '(', ')', 'const', 'override', '{', 'class0', '::', 'var10', '(', ')', ';', 'std', '::', 'cout', '<<', 'lit5', '<<', 'var8', '<<', 'lit4', ';', '}', '}', ';', 'int', 'func1', '(', 'int', 'var11', ')', '{', 'std', '::', 'cout', '<<', 'lit6', '<<', 'var11', '<<', 'lit4', ';', 'return', 'var11', '*', 'var11', ';', '}', 'int', 'main', '(', ')', '{', 'int', 'var12', '=', 'func1', '(', 'lit7', ')', ';', 'std', '::', 'cout', '<<', 'lit8', '<<', 'var12', '<<', 'lit4', ';', 'std', '::', 'vector', '<', 'class0', '*', '>', 'var13', ';', 'var13', '.', 'func2', '(', 'new', 'class1', '(', 'lit9', ',', 'lit10', ',', 'lit11', ',', 'lit12', ',', 'lit13', ')', ')', ';', 'var13', '.', 'func2', '(', 'new', 'class0', '(', 'lit14', ',', 'lit15', ',', 'lit16', ',', 'lit17', ')', ')', ';', 'for', '(', 'const', 'auto', '&', 'var14', ':', 'var13', ')', '{', 'var14', '->', 'var10', '(', ')', ';', 'std', '::', 'cout', '<<', 'lit18', ';', '}', 'for', '(', 'auto', 'var14', ':', 'var13', ')', 'delete', 'var14', ';', 'return', 'lit19', ';', '}']

---- Variable Dict ----
{'number_plate': 'var0', 'model_name': 'var1', 'price': 'var2', 'year_of_manufacture': 'var3', 'num': 'var4', 'model': 'var5', 'p': 'var6', 'year': 'var7', 'num_doors': 'var8', 'doors': 'var9', 'printInfo': 'var10', 'x': 'var11', 'r': 'var12', 'vehicles': 'var13', 'v': 'var14'}

---- Function Dict ----
{'printInfo': 'func0', 'square': 'func1', 'push_back': 'func2'}

---- Literal Dict ----
{'"Vehicle: "': 'lit0', '", Model: "': 'lit1', '", Price: $"': 'lit2', '", Year: "': 'lit3', '"\\n"': 'lit4', '"Number of doors: "': 'lit5', '"Squaring the number: "': 'lit6', '5': 'lit7', '"Result: "': 'lit8', '"ABC123"': 'lit9', '"Tesla Model S"': 'lit10', '80000': 'lit11', '2022': 'lit12', '4': 'lit13', '"XYZ789"': 'lit14', '"Generic Vehicle"': 'lit15', '30000': 'lit16', '2015': 'lit17', '"--------------------\\n"': 'lit18', '0': 'lit19'}

---- Struct Dict ----
{}

---- Class Dict ----
{'Vehicle': 'class0', 'Car': 'class1'}
Final output written to cleaned_simple.cpp
```

