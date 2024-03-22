#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <ctime>

using dmatrix = std::vector<std::vector<double>>;
using drow = std::vector<double>;

double random(const double& min, const double& max) {
    std::mt19937_64 rng{}; rng.seed(std::random_device{}());
    return std::uniform_real_distribution<>{min, max}(rng);
}

//Transpose matrix func 
dmatrix T(const dmatrix& m) noexcept {
    dmatrix mat;
    for(size_t i = 0; i < m[0].size(); i++) {
        mat.push_back({});
        for(size_t j = 0; j < m.size(); j++) mat[i].push_back(m[j][i]);
    }
    return mat;  
}

//Matrix multiplication
dmatrix operator*(const dmatrix& m1, const dmatrix& m2) noexcept {
    dmatrix m3 = T(m2);
    dmatrix result;
    for (size_t i=0; i<m1.size(); i++){
        for (size_t j=0; j<m3.size(); j++){
            double dot_product = 0.0;
            for(size_t k=0; k<m3[0].size(); k++) dot_product += m1[i][j] * m3[j][k];
            result[i].push_back(dot_product);
        }
    }
    return result;
}