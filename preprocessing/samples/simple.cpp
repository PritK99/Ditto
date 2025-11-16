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
