#include <iostream>
#include <vector>
#include <string>

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
    virtual ~Vehicle() = default;
};
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
int square(int x) {
    std::cout << "Squaring the number: " << x << "\n";
    return x * x;
}
int main() {
    int r = square(5);
    std::cout << "Result: " << r << "\n";
    std::vector<Vehicle*> vehicles;
    vehicles.push_back(new Car("ABC123", "Tesla Model S", 80000, 2022, 4));
    vehicles.push_back(new Vehicle("XYZ789", "Generic Vehicle", 30000, 2015));
    for (const auto& v : vehicles) {
        v->printInfo();
        std::cout << "--------------------\n";
    }
    for (auto v : vehicles) delete v;
    return 0;
}