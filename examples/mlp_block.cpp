#include "modules.hpp"


int main(int argc, char** argv)
{
    MLPBlock mlp(2, 3, std::vector<int64_t>{64, 64});
    std::cout << mlp << std::endl;
    return 0;
}
