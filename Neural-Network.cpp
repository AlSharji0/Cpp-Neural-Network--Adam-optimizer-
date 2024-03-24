#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

using dmatrix = std::vector<std::vector<double>>;
using drow = std::vector<double>;

double random(const double& min, const double& max){
    std::mt19937_64 rng{}; rng.seed(std::random_device{}());
    return std::uniform_real_distribution<>{min, max}(rng);
}

//Transpose matrix func 
dmatrix T(const dmatrix& m) noexcept {
    dmatrix mat;
    for(size_t i = 0; i < m[0].size(); i++){
        mat.push_back({});
        for(size_t j = 0; j < m.size(); j++) mat[i].push_back(m[j][i]);
    }
    return mat;  
}

//Matrix multiplication
dmatrix operator*(const dmatrix& m1, const dmatrix& m2) noexcept{
    dmatrix m3 = T(m2);
    dmatrix result;
    for(size_t i=0; i<m1.size(); i++){
        for (size_t j=0; j<m3.size(); j++){
            double dot_product = 0.0;
            for(size_t k=0; k<m3[0].size(); k++) dot_product += m1[i][j] * m3[j][k];
            result[i].push_back(dot_product);
        }
    }
    return result;
}

//Matrix addition
dmatrix operator+ (const dmatrix& m, const drow& drow) noexcept{
    dmatrix result{};
    for(size_t i=0; i<m.size(); i++){
        result.push_back({});
        for(size_t j=0; j<m[0].size(); j++){
            result[i].push_back(m[i][j] + drow[i]);
        }
    }
}

//Output
std::ostream& operator<<(std::ostream& os,const dmatrix& dm) noexcept {
    for(auto& row : dm){
        for(auto& item : row) os << item << " ";
        os << "\n";
    }return os;
}


class DenseLayer {
private:
    dmatrix m_weights, m_outputs;
    drow biases;
public:
    DenseLayer(const size_t& n_inputs, const size_t& n_neurons) : m_weights(n_inputs, drow(n_neurons)), biases(n_neurons, 0) {
        for(size_t i=0; i<n_inputs; i++){
            for(size_t j=0; j<n_neurons; j++) m_weights[i][j] = random(-1., 1.);
        }
    }
    dmatrix forward(const dmatrix& inputs){
        m_outputs = (inputs * m_weights) + biases;
        return m_outputs;
    }
};

class ReluActivation{
private:
    dmatrix output;
public:
    dmatrix forward(const dmatrix& inputs) {
        output = dmatrix(inputs.size(), drow(inputs[0].size(), 0.));
        for(size_t i = 0; i < inputs.size(); i++) {
            for(size_t j = 0; j < inputs[0].size(); j++) output[i][j] = std::max(0., inputs[i][j]);
        }return output;
    }
};


class SoftMaxActivation{
private:
    dmatrix output;
public:
    dmatrix forward(const dmatrix& inputs){
        output = dmatrix(inputs.size(), drow(inputs[0].size(), 0.));
        for(size_t i=0; i<inputs.size(); i++){
            double max_val = *std::max_element(inputs[i].begin(), inputs[i].end());
            double exp_sum = 0.;
            for(size_t j=0; j<inputs[0].size(); j++){
                double exp_val = std::exp(inputs[i][j]-max_val);
                output[i][j] = exp_val;
                exp_sum += exp_val;
            }
            for(size_t j=0; j<inputs[0].size(); j++) output[i][j] /= exp_sum;
        }
    }
};
