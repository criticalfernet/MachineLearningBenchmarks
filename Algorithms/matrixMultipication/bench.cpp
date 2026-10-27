#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

using Matrix = Eigen::MatrixXd;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

void multiply(const Matrix& A, const Matrix& B, Matrix& C);

struct TestCase {
    Matrix A;
    Matrix B;
    Matrix C;
};
struct Statistics {
    double median_ms;
    double mean_ms;
    double min_ms;
    double max_ms;
};
struct TestResult {
    int test_case;
    Statistics stats;
};

constexpr int WARMUPS = 10;
constexpr int REPETITIONS = 20;
constexpr double RTOL = 1e-10;
constexpr double ATOL = 1e-10;

const fs::path ROOT = fs::current_path();
const fs::path DATASET_DIR = ROOT / "datasets" / "matrixMultiplication";
const fs::path RESULT_FILE = ROOT / "results" / "matrixMultiplication_cpp.csv";

Statistics calculate_statistics(std::vector<double> values) {
    const std::size_t n = values.size();
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double median_ms;

    if (n % 2 == 0) {
        median_ms = (values[n / 2 - 1] + values[n / 2]) / 2.0;
    } else {
        median_ms = values[n / 2];
    }

    return {
        median_ms,
        sum / static_cast<double>(n),
        values.front(),
        values.back()
    };
}

std::string trim(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";

    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

Matrix read_matrix(std::ifstream& file, int n) {
    Matrix M(n, n);
    for (int i = 0; i < n; ++i) {
        std::string line;
        std::getline(file, line);

        std::stringstream ss(line);

        for (int j = 0; j < n; ++j) {
            std::string value;
            std::getline(ss, value, ',');
            M(i, j) = std::stod(trim(value));
        }
    }

    return M;
}

bool find_next_test(std::ifstream& file) {
    std::string line;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.rfind("# Test", 0) == 0) { return true;}
    }

    return false;
}

TestCase read_test(std::ifstream& file, int n) {
    TestCase test;

    std::string header;
    std::getline(file, header);
    test.A = read_matrix(file, n);

    std::getline(file, header);
    test.B = read_matrix(file, n);

    std::getline(file, header);
    test.C = read_matrix(file, n);

    return test;
}

int read_header(std::ifstream& file) {
    std::string line;

    std::getline(file, line);

    const int num_tests = std::stoi(trim(line));

    std::getline(file, line);

    const int n = std::stoi(trim(line));
    return n;
}

bool matrices_equal(const Matrix& actual,const Matrix& expected) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        return false;
    }

    for (Eigen::Index i = 0; i < actual.rows(); ++i) {
        for (Eigen::Index j = 0; j < actual.cols(); ++j) {
            const double a = actual(i, j);
            const double b = expected(i, j);

            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (a != b) return false;
                continue;
            }

            if (std::abs(a - b) > ATOL + RTOL * std::abs(b)) {
                return false;
            }
        }
    }

    return true;
}

TestResult benchmark_test( const TestCase& test,int test_number) {
    int n = test.A.rows(); 
    Eigen::setNbThreads(n >= 768 ? 8 : (n >= 384 ? 4 : 1)); 
    double consume = 0.0;
    for (int i = 0; i < WARMUPS; ++i) {
       Matrix result;
       multiply(test.A, test.B, result);
       consume += result(0, 0);
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(REPETITIONS);

    for (int i = 0; i < REPETITIONS; ++i) {
        double consume = 0.0;
        Matrix result;
        const auto start = Clock::now();
        multiply(test.A, test.B, result);
        const auto end = Clock::now();
        consume += result(0, 0);

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        if (!matrices_equal(result, test.C)) {
            throw std::runtime_error("Correctness check failed: test " + test_number);
        }

        samples_ms.push_back(elapsed_ms);
    }

    return {
        test_number,
        calculate_statistics(samples_ms)
    };
}

void write_csv_row( std::ofstream& file, int matrix_size,const std::string& test_case, const Statistics& stats) {
    file << "C++,Matrix Multiplication,"
         << matrix_size << ','
         << test_case << ','
         << stats.median_ms << ','
         << stats.mean_ms << ','
         << stats.min_ms << ','
         << stats.max_ms << ','
         << WARMUPS << ','
         << REPETITIONS << '\n';
}

void save_results(const std::vector<TestResult>& results, const Statistics& summary, int matrix_size) {
    fs::create_directories(RESULT_FILE.parent_path());

    const bool needs_header = !fs::exists(RESULT_FILE) || fs::file_size(RESULT_FILE) == 0;

    std::ofstream file(RESULT_FILE, std::ios::app);

    if (!file) {
        throw std::runtime_error(
            "Could not open result file: " + RESULT_FILE.string()
        );
    }

    file << std::fixed << std::setprecision(6);

    if (needs_header) {
        file << "implementation,algorithm,matrix_size,test_case,"
                "median_latency_ms,mean_latency_ms,min_latency_ms,"
                "max_latency_ms,warmup_runs,repetitions\n";
    }

    for (const TestResult& result : results) {
        write_csv_row(
            file,
            matrix_size,
            std::to_string(result.test_case),
            result.stats
        );
    }

    write_csv_row(file, matrix_size, "SUMMARY", summary);
    std::cout << "Results saved to: " << RESULT_FILE.string() << '\n';
}

void benchmark_dataset(const fs::path& path) {
    std::cout << "\nLoading dataset: " << path.filename().string() << '\n';

    std::ifstream file(path);
    std::string line;

    std::getline(file, line);

    const int num_tests = std::stoi(trim(line));

    std::getline(file, line);
    const int n = std::stoi(trim(line));

    std::cout << "Matrix size: " << n << '\n' << "Test cases: " << num_tests << '\n';

    std::vector<TestResult> results;
    std::vector<double> timings;

    results.reserve(num_tests);
    timings.reserve(num_tests);

    for (int i = 1; i <= num_tests; i++) {
        if (!find_next_test(file)) {
            throw std::runtime_error(
                "Expected test " + std::to_string(i) +
                " but reached end of file"
            );
        }

        const TestCase test = read_test(file, n);

        const TestResult result = benchmark_test(test, i);

        timings.push_back(result.stats.median_ms);
        results.push_back(result);

        std::cout << "Test " << i
                  << ": median = "
                  << std::fixed << std::setprecision(6)
                  << result.stats.median_ms << " ms\n";
    }

    if (results.empty()) {
        std::cout << "No test cases found.\n";
        return;
    }

    const Statistics summary = calculate_statistics(timings);

    std::cout << "\n--- Benchmark Completed ---\n";

    save_results(results, summary, n);
}

int main() {
    try {
        if (!fs::exists(DATASET_DIR) || !fs::is_directory(DATASET_DIR)) {
            throw std::runtime_error("Dataset directory not found: ");
        }

        std::vector<fs::path> datasets;

        for (const auto& entry : fs::directory_iterator(DATASET_DIR)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string filename = entry.path().filename().string();

            if (filename.rfind("matrix_", 0) == 0 && entry.path().extension() == ".txt") {
                datasets.push_back(entry.path());
            }
        }

        std::sort(
            datasets.begin(),
            datasets.end(),
            [](const fs::path& a, const fs::path& b) {
                const auto size_a = std::stoull(
                    a.stem().string().substr(7)
                );

                const auto size_b = std::stoull(
                    b.stem().string().substr(7)
                );

                return size_a < size_b;
            }
        );

        for (const auto& dataset : datasets) {
            benchmark_dataset(dataset);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}