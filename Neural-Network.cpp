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

dmatrix generateSpiralData(int samples_per_class, int classes) {
    dmatrix data;
    double delta_theta = 2.0 * M_PI / classes;

    for (int c = 0; c < classes; c++){
        dmatrix spiral_class;
        double r = 0.0;
        double theta = delta_theta * c;
        for (int i = 0; i < samples_per_class; ++i) {
            double dr = 5.0 * (1.0 * i / samples_per_class);
            double x = r * cos(theta);
            double y = r * sin(theta);
            spiral_class.push_back({x, y});
            r += dr;
            theta += 0.5;
        }
        data.insert(data.end(), spiral_class.begin(), spiral_class.end());
    } return data;
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
dmatrix operator+ (const dmatrix& m, const drow& row) noexcept{
    dmatrix result{};
    for(size_t i=0; i<m.size(); i++){
        result.push_back({});
        for(size_t j=0; j<m[0].size(); j++){
            result[i].push_back(m[i][j] + row[i]);
        }
    } return result;
}

// Output
std::ostream& operator<<(std::ostream& os,const dmatrix& dm) noexcept {
    for(auto& row : dm){
        for(auto& item : row) os << item << " ";
        os << "\n";
    }return os;
}

//For jacobian matrix (Kronecker delta)
dmatrix eye(size_t n){
    dmatrix identity(n, drow(n, 0.));
    for(size_t i=0; i<n; i++){
        identity[i][i]=1.;
    } return identity;
}

struct DenseLayer {
    dmatrix weight_momentums;
    dmatrix weight_cache;
    drow bias_momentums;
    drow bias_cache;
    dmatrix m_weights, m_outputs;
    drow biases;
    dmatrix m_inputs;
    dmatrix dweights;
    dmatrix dinputs;
    drow dbiases;

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
        dweights = T(m_inputs) * dvalue;
        dinputs = T(m_weights) * dvalue;
        dbiases = std::accumulate(dvalue.begin(), dvalue.end(), drow(dvalue[0].size(), 0),
    [](const drow& acc, const drow& val) {
        drow result;
        std::transform(acc.begin(), acc.end(), val.begin(), std::back_inserter(result), std::plus<double>());
        return result;
    });
        return dinputs;
    }

    // Helper function for optimizer
    void initialize_momentum_cache(){
        weight_momentums.resize(m_weights[0].size(), drow(m_weights[0].size(), 0.));
        weight_cache.resize(m_weights[0].size(), drow(m_weights[0].size(), 0.));
        bias_momentums = drow(m_weights[0].size(), 0.);
        bias_cache = drow(m_weights[0].size(), 0.);
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
    dmatrix backward(const dmatrix& dvalues){
        kdelta = eye(dinputs.size());
        dinputs = dmatrix(dvalues.size(), drow(dvalues.size(), 0.));
        jacobian = dmatrix(dvalues.size(), drow(dvalues.size(), 0.));
        for(size_t i=0; i<dinputs.size(); i++){
            for(size_t j=0; j<dinputs[0].size(); j++){
                kdelta[i][j] = kdelta[i][j] * output[i][j];
                jacobian[i][j] = kdelta[i][j] - (output[i][j] * output[i][j]);
                dinputs[i][j] = jacobian[i][j] * dvalues[i][j];
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
    virtual drow forward(dmatrix& predictions, const dmatrix& ytrue) = 0;
    double calculate_data_loss(dmatrix& output, const dmatrix& y){
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
    dmatrix dinputs;
public:
    drow forward(dmatrix& predictions, const dmatrix& ytrue){
        clipMatrix(predictions, 1e-7, 1. - 1e-7);
        for(size_t i=0; i<predictions.size(); i++){
            correctp.push_back({});
            for(size_t j=0; j<predictions[0].size(); j++) correctp[i].push_back(predictions[i][j] * ytrue[i][j]);
            nlogp[i] = std::accumulate(correctp[i].begin(), correctp[i].end(), 0.);
            nlogp[i] = -std::log(nlogp[i]);
        } return nlogp;
    }
    dmatrix backward(const dmatrix& dvalues, const dmatrix& ytrue){
            size_t samples = dvalues.size();
            for(size_t i=0; i<samples; i++){
                for(size_t j=0; j<ytrue[0].size(); j++) dinputs[i].push_back(-ytrue[i][j] / dvalues[i][j]);
                double sum = std::accumulate(dinputs[i].begin(), dinputs[i].end(), 0.0);
                for(size_t j=0; j<dinputs[0].size(); j++) dinputs[i].push_back(dinputs[i][j] / samples);
            } return dinputs;
        }
};

class AdamOptimizer{
private:
    double learning_rate = 0.001;
    double current_learning_rate;
    double decay = 0.;
    int iterations;
    double epsilon = 1e-7;
    double beta_1 = 0.9;
    double beta_2 = 0.999;
    dmatrix weight_momentums_corrected;
    drow bias_momentums_corrected;
    dmatrix weight_cache_corrected;
    drow bias_cache_corrected;    

public:
    void pre_update(){
        if(decay) current_learning_rate = learning_rate * (1. / (1. + decay * iterations));
    }

    void update(DenseLayer& DenseLayer){
        if(DenseLayer.weight_cache.empty()) DenseLayer.initialize_momentum_cache();
        for(size_t i=0; i<DenseLayer.m_weights.size(); i++){
            for(size_t j=0; j<DenseLayer.m_weights[0].size(); j++){
                DenseLayer.weight_momentums[i][j] = beta_1 * DenseLayer.weight_momentums[i][j] + (1 - beta_1) * DenseLayer.dweights[i][j];
                DenseLayer.bias_momentums[i] = beta_1 * DenseLayer.bias_momentums[i] + (1 - beta_1) * DenseLayer.bias_momentums[i];
                weight_momentums_corrected[i][j] = DenseLayer.weight_momentums[i][j] / (1 - std::pow(beta_1, iterations+1));
                bias_momentums_corrected[i] = DenseLayer.bias_momentums[i] / (1 - std::pow(beta_1, iterations+1));
                DenseLayer.weight_cache[i][j] = beta_2 * DenseLayer.weight_cache[i][j] + (1 - beta_2) * std::pow(DenseLayer.dweights[i][j], 2);
                DenseLayer.bias_cache[i] = beta_2 * DenseLayer.bias_cache[i] + (1 - beta_2) * std::pow(DenseLayer.dbiases[i], 2);
                weight_cache_corrected[i][j] = DenseLayer.weight_cache[i][j] / (1 - std::pow(beta_2, iterations+1));
                bias_cache_corrected[i] = DenseLayer.bias_cache[i] / (1 - std::pow(beta_2, iterations + 1));
                DenseLayer.m_weights[i][j] += -current_learning_rate * weight_cache_corrected[i][j] / (std::sqrt(weight_cache_corrected[i][j]) + epsilon);
                DenseLayer.biases[i] += -current_learning_rate * bias_momentums_corrected[i] / (std::sqrt(bias_momentums_corrected[i]) + epsilon);
            }
        }
    }

    void post_update(){
        iterations += 1;
    }
};

int main() {
    // Generate data
    dmatrix data = generateSpiralData(100, 3);

    // Extract features and labels
    dmatrix X, y;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i % 2 == 0) {
            X.push_back(data[i]);
        } else {
            y.push_back(data[i]);
        }
    }

    // Convert labels to one-hot encoding
    dmatrix y_true;
    for (size_t i = 0; i < y.size(); ++i) {
        drow one_hot(3, 0.0);
        int label = static_cast<int>(y[i][0]);
        one_hot[label] = 1.0;
        y_true.push_back(one_hot);
    }

    // Initialize Neural Network layers
    DenseLayer layer1(2, 64);
    ReluActivation activation1;
    DenseLayer layer2(64, 3);
    SoftMaxActivation activation2;
    Loss_categoricalCrossentropy loss;
    dmatrix dvalue = activation2.forward(layer2.forward(activation1.forward(layer1.forward(X))));

    // Train the neural network
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        double current_loss = loss.calculate_data_loss(dvalue, y_true);
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << current_loss << std::endl;
        }

        // Backward pass
        dmatrix dvalues = loss.backward(dvalue, y_true);
        layer1.backward(activation1.backward(layer2.backward(activation1.backward(activation2.backward(dvalue)))));

        // Update weights and biases
        AdamOptimizer optimizer;
        optimizer.pre_update();
        optimizer.update(layer1);
        optimizer.update(layer2);
        optimizer.post_update();
    }
}