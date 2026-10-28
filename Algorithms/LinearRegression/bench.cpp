#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

constexpr int WARMUPS = 3;
constexpr int REPETITIONS = 10;
constexpr int FEATURES = 512;

void linear_regression(const Matrix& X, const Vector& y, Vector& C);

const std::filesystem::path DATASET_FILE = "datasets/LinearRegression/train.csv";
const std::filesystem::path RESULT_FILE = "results/LinearRegression_cpp.csv";

void load_dataset(const std::filesystem::path& path, Matrix& X, Vector& y) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open dataset: " + path.string());
    }

    std::vector<double> all_x;
    std::vector<double> all_y;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(file, line)) {
        line_number++;
        if (line.empty()) continue;

        std::vector<double> values;
        std::stringstream ss(line);
        std::string field;

        while (std::getline(ss, field, ',')) {
            try {
                const double value = std::stod(field);
                values.push_back(value);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid value on line " + std::to_string(line_number));
            }
        }

        if (values.size() != FEATURES + 1) {
            throw std::runtime_error("mismatch features");
        }

        all_x.insert(all_x.end(), values.begin(), values.begin() + FEATURES);
        all_y.push_back(values[FEATURES]);
    }

    if (all_y.empty()) throw std::runtime_error("Dataset is empty.");

    const Eigen::Index rows = static_cast<Eigen::Index>(all_y.size());
    X.resize(rows, FEATURES);
    y.resize(rows);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < FEATURES; ++j) {
            X(i, j) = all_x[static_cast<std::size_t>(i) * FEATURES + j];
        }
        y(i) = all_y[static_cast<std::size_t>(i)];
    }
}

struct Statistics {
    double median_ms;
    double mean_ms;
    double min_ms;
    double max_ms;
};

void save_results(Statistics stat, Eigen::Index rows, Eigen::Index features) {
    std::filesystem::create_directories(RESULT_FILE.parent_path());
    std::ofstream file(RESULT_FILE, std::ios::out | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Could not open results file");
    }

    file << "implementation,algorithm,matrix_size,test_case,"
            "median_latency_ms,mean_latency_ms,min_latency_ms,max_latency_ms,"
            "warmup_runs,repetitions\n";

    file << std::fixed << std::setprecision(6)
         << "C++,Linear Regression," << rows << "x" << features
         << ",SUMMARY," << stat.median_ms << "," << stat.mean_ms << ","
         << stat.min_ms << "," << stat.max_ms << ","
         << WARMUPS << "," << REPETITIONS << '\n';

    std::cout << "\nResults saved to: " << RESULT_FILE << '\n';
}


int main() {
    try {
        Matrix X;
        Vector y;
        load_dataset(DATASET_FILE, X, y);

        std::println("Dataset shape: X=({}, {}), Y=({})",X.rows(), X.cols(), y.size());

        double consume = 0.0;
        Eigen::setNbThreads(8);
        for (int i = 0; i < WARMUPS; ++i) {
            Vector beta;
            linear_regression(X, y, beta);
            consume += beta(0,0);
        }

        std::vector<double> timings_ms;
        timings_ms.reserve(REPETITIONS);
        
        for (int i = 0; i < REPETITIONS; ++i) {
            Vector beta;
            const auto start = Clock::now();
            linear_regression(X, y, beta);
            const auto end = Clock::now();
            consume += beta(0,0);

            if (!beta.allFinite()) {
                throw std::runtime_error("Regression returned non-finite coefficients.");
            }

            timings_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::sort(timings_ms.begin(), timings_ms.end());
        const double min_ms = timings_ms.front();
        const double max_ms = timings_ms.back();
        const double median_ms = (timings_ms[REPETITIONS / 2 - 1] + timings_ms[REPETITIONS / 2]) / 2.0;

        double sum = 0.0;
        for (double t : timings_ms) sum += t;
        const double mean_ms = sum / timings_ms.size();

        std::cout << "\n--- Benchmark Completed ---\n";

        Statistics stat = {median_ms, mean_ms, min_ms, max_ms};

        save_results(stat, X.rows(), X.cols());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}