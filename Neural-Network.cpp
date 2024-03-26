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

// Transpose matrix func 
dmatrix T(const dmatrix& m) noexcept {
    dmatrix mat;
    for(size_t i = 0; i < m[0].size(); i++){
        mat.push_back({});
        for(size_t j = 0; j < m.size(); j++) mat[i].push_back(m[j][i]);
    }
    return mat;  
}

// Matrix multiplication
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

// Matrix addition
dmatrix operator+ (const dmatrix& m, const drow& drow) noexcept{
    dmatrix result{};
    for(size_t i=0; i<m.size(); i++){
        result.push_back({});
        for(size_t j=0; j<m[0].size(); j++){
            result[i].push_back(m[i][j] + drow[i]);
        }
    }
}

// Output
std::ostream& operator<<(std::ostream& os,const dmatrix& dm) noexcept {
    for(auto& row : dm){
        for(auto& item : row) os << item << " ";
        os << "\n";
    }return os;
}

//For jacobian matrix (Kronecker delta)
dmatrix eye(int n){
    dmatrix identity(n, drow(n, 0.));
    for(size_t i=0; i<n; i++){
        identity[i][i]=1.;
    } return identity;
}

class DenseLayer {
private:
    dmatrix m_weights, m_outputs;
    drow biases;
    dmatrix m_inputs;
public:
    DenseLayer(const size_t& n_inputs, const size_t& n_neurons) : m_weights(n_inputs, drow(n_neurons)), biases(n_neurons, 0) {
        for(size_t i=0; i<n_inputs; i++){
            for(size_t j=0; j<n_neurons; j++) m_weights[i][j] = random(-1., 1.);
        }
    }
    dmatrix forward(const dmatrix& inputs){
        m_inputs = inputs;
        m_outputs = (inputs * m_weights) + biases;
        return m_outputs;
    }

    dmatrix backward(const dmatrix& dvalue){
        dmatrix dweights = T(m_inputs) * dvalue;
        dmatrix dinputs = T(m_weights) * dvalue;
        drow dbiases = std::accumulate(dvalue.begin(), dvalue.end(), drow(dvalue[0].size(), 0));
    }
};

class ReluActivation{
private:
    dmatrix output;
    dmatrix dinputs;
public:
    dmatrix forward(const dmatrix& inputs) {
        output = dmatrix(inputs.size(), drow(inputs[0].size(), 0.));
        for(size_t i = 0; i < inputs.size(); i++) {
            for(size_t j = 0; j < inputs[0].size(); j++) output[i][j] = std::max(0., inputs[i][j]);
        }return output;
    }
    dmatrix backward(const dmatrix& dvalue){
        dinputs = dmatrix(output.size(), drow(output[0].size(), 0.));
        for(size_t i=0; i<dvalue.size(); i++){
            for(size_t j=0; j<dvalue[0].size(); j++){
                if(output[i][j] <= 0) dinputs[i][j] = 0;
                else dinputs[i][j] = dvalue[i][j];
            }
        } return dinputs;
    }
};


class SoftMaxActivation{
private:
    dmatrix output;
    dmatrix dinputs;
    dmatrix kdelta;
    dmatrix jacobian;
public:
    dmatrix forward(const dmatrix& inputs){
        output = dmatrix(inputs.size(), drow(inputs[0].size(), 0.));
        for(size_t i=0; i<inputs.size(); i++){
            double max_val = *std::max_element(inputs[i].begin(), inputs[i].end());
            double exp_sum = 0.;
            for(size_t j=0; j<inputs[0].size(); j++){
                double exp_val = std::exp(inputs[i][j] - max_val);
                output[i][j] = exp_val;
                exp_sum += exp_val;
            }
            for(size_t j=0; j<inputs[0].size(); j++) output[i][j] /= exp_sum;
        } return output;
    }
    dmatrix backward(drow& dvalues){
        kdelta = eye(dinputs.size());
        dinputs = dmatrix(dvalues.size(), drow(dvalues.size(), 0.));
        jacobian = dmatrix(dvalues.size(), drow(dvalues.size(), 0.));
        for(size_t i=0; i<dinputs.size(); i++){
            for(size_t j=0; j<dinputs[0].size(); j++){
                kdelta[i][j] = kdelta[i][j] * output[i][j];
                jacobian[i][j] = kdelta[i][j] - (output[i][j] * output[i][j]);
                dinputs[i][j] = jacobian[i][j] * dvalues[j];
            }
        } return dinputs;
    }
    
};

// Clip softmax output values
void clipMatrix(dmatrix& input, double min, double max){
    for(size_t i=0; i<input.size(); i++){
        for(size_t j=0; j<input[0].size(); j++) input[i][j] = std::max(min, std::min(input[i][j], max));
    }
}

struct loss{
    virtual drow forward(dmatrix& predictions, const drow& ytrue) = 0;
    double calculate_data_loss(dmatrix& output, const drow& y){
        drow sample_loss = forward(output, y);
        double total_loss =0.;
        for(double loss:sample_loss) total_loss += loss;
        double average_loss = total_loss / sample_loss.size();
        return average_loss;
    }
};

class Loss_categoricalCrossentropy: public loss {
private:
    dmatrix correctp;
    drow nlogp;
    dmatrix inputs;
    size_t samples;
    size_t labels;
    drow dinputs;
public:
    drow forward(dmatrix& predictions, const drow& ytrue){
        clipMatrix(predictions, 1e-7, 1. - 1e-7);
        for(size_t i=0; i<predictions.size(); i++){
            correctp.push_back({});
            for(size_t j=0; j<predictions[0].size(); j++) correctp[i].push_back(predictions[i][j] * ytrue[j]);
            nlogp[i] = std::accumulate(correctp[i].begin(), correctp[i].end(), 0.);
            nlogp[i] = -std::log(nlogp[i]);
        } return nlogp;
    }
    drow backward(const drow& dvalues, const drow& ytrue){
            size_t samples = dvalues.size();
            for(size_t i=0; i<samples; i++){
                dinputs.push_back(-ytrue[i]/dvalues[i]);
                dinputs[i] = dinputs[i]/samples;
            } return dinputs;
        }
};
