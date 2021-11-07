#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#define N 500
#define N_MEASUREMENTS 100

using namespace std;
using matrix = vector<vector<int>>;

void read_matrix(matrix &m) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      cin >> m[i][j];
    }
  }
}

matrix multiply(const matrix &A, const matrix &B) {
  auto matrixR = matrix(N, vector<int>(N));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        matrixR[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return matrixR;
}

int main(void) {
  ofstream fresult("result.csv", ofstream::out);

  if (!fresult.is_open()) {
    cerr << "file did not open\n";
    return 1;
  }

  fresult << "time,\n";

  auto matrixA = matrix(N, vector<int>(N));
  auto matrixB = matrix(N, vector<int>(N));

  read_matrix(matrixA);
  read_matrix(matrixB);

  clog << "terminou leitura\n";

  for (int i = 0; i < N_MEASUREMENTS; i++) {
    if (i > 0) {
      fresult << ',' << endl;
    }

    const auto start = chrono::high_resolution_clock::now();
    multiply(matrixA, matrixB);
    const auto end = chrono::high_resolution_clock::now();

    const chrono::duration<double> duration = end - start;

    fresult << duration.count();
  }

  fresult.close();

  return 0;
}