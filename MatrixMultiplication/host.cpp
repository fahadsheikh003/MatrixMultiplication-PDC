#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "CL/cl.hpp"

using namespace std;

#define SIZE 512

cl::Device* getDevice() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> allDevices;

    for (auto& p : platforms) {
        vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        for (auto& d : devices) {
            allDevices.push_back(d);
        }

        devices.clear();

        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        for (auto& d : devices) {
            allDevices.push_back(d);
        }
    }

    int choice = -1;
    do {
        cout << "Select a device to use from the following list.." << endl << endl;
        for (int i = 0; i < allDevices.size(); i++) {
            auto& d = allDevices[i];
            cout << "Device #: " << i + 1 << endl;
            cout << "Device Name: " << d.getInfo<CL_DEVICE_NAME>() << endl;
            cout << "Device Type: " << (d.getInfo<CL_DEVICE_TYPE>() == 2 ? "CPU" : "GPU") << endl << endl;
        }
        cout << "Enter device #: ";
        cin >> choice;
        cin.ignore();
        cout << endl;
    } while (choice < 1 || choice > allDevices.size());

    return new cl::Device(allDevices[choice - 1]);
}

void initializeMatrix(int* matrix) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i * SIZE + j] = rand() % 10;
        }
    }
}

void matrixMultiplication(int* A, int* B, int* C) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            C[i * SIZE + j] = 0;
            for (int k = 0; k < SIZE; ++k) {
                C[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
            }
        }
    }
}

int main() {
    srand(time(0));
    int* A, * B, * C;
    A = new int[SIZE * SIZE]; 
    B = new int[SIZE * SIZE];
    C = new int[SIZE * SIZE] {0};
    cout << "Initializing Matrices" << endl;
    initializeMatrix(A);
    initializeMatrix(B);
    cout << "Matrices Initialized" << endl << endl;

    cl::Device* device = getDevice();
    cout << "Selected Device: " << device->getInfo<CL_DEVICE_NAME>() << endl;

    ifstream file("kernel.cl");
    string src(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(*device);
    cl::Program program(context, sources);
    
    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device)
            << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << std::endl;
        exit(1);
    }

    int actualSizeOfMatrix = SIZE * SIZE * sizeof(int);
    cl::Buffer bufA(context, CL_MEM_READ_ONLY, actualSizeOfMatrix);
    cl::Buffer bufB(context, CL_MEM_READ_ONLY, actualSizeOfMatrix);
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, actualSizeOfMatrix);

    cl::Kernel kernel(program, "matrixMultiplication");
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, SIZE);

    cl::CommandQueue queue(context, *device, CL_QUEUE_PROFILING_ENABLE);

    queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, actualSizeOfMatrix, A);
    queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, actualSizeOfMatrix, B);

    cl::NDRange globalWorkSize(SIZE, SIZE);
    cl::NDRange localWorkSize(16, 16); // factor of SIZE

    cl::Event _event;

    auto start = chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, localWorkSize, NULL, &_event);
    _event.wait();

    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, actualSizeOfMatrix, C);
    queue.finish();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> openclTime = end - start;

    cout << "Elapsed Time in Parallel Computation: " << fixed << openclTime.count() << " s" << endl;

    //cl_ulong startTime = _event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //cl_ulong endTime = _event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    //cl_ulong elapsedTime = endTime - startTime;

    //cout << "Elapsed Time in Parallel Computation: " << elapsedTime << " ns" << endl;

    //for (int i = 0; i < SIZE; i++) {
    //    for (int j = 0; j < SIZE; j++) {
    //        cout << C[i * SIZE + j] << " ";
    //    }
    //    cout << endl;
    //}
    
    int* localC = new int[SIZE * SIZE] {0};
    start = chrono::high_resolution_clock::now();
    matrixMultiplication(A, B, localC);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> serialTime = end - start;

    cout << "Elapsed Time in Serial Computation: " << fixed << serialTime.count() << " s" << endl;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (C[i * SIZE + j] != localC[i * SIZE + j]) {
                cerr << "Error found! at index [" << i << "," << j << "]" << endl;
                cerr << "Expected Value: " << localC[i * SIZE + j] << " Value Found: " << C[i * SIZE + j] << endl;
                exit(1);
            }
        }
    }

    cout << "Local Matrix Multiplication `matrixMultiplication` and OpenCL Matrix Multiplication `matrixMultiplication` match!" << endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] localC;

    return 0;
}


// Results for Graph Generation
/*
vector<cl::Device> getAllDevices() {
    vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	vector<cl::Device> allDevices;

    for (auto& p : platforms) {
		vector<cl::Device> devices;
		p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        for (auto& d : devices) {
			allDevices.push_back(d);
		}

		devices.clear();

		p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        for (auto& d : devices) {
			allDevices.push_back(d);
		}
	}

	return allDevices;
}

double runParallel(cl::Device* device, int* A, int *B, int* localC) {
    int* C = new int[SIZE * SIZE] {0};

    ifstream file("kernel.cl");
    string src(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(*device);
    cl::Program program(context, sources);

    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device)
            << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << std::endl;
        exit(1);
    }

    int actualSizeOfMatrix = SIZE * SIZE * sizeof(int);
    cl::Buffer bufA(context, CL_MEM_READ_ONLY, actualSizeOfMatrix);
    cl::Buffer bufB(context, CL_MEM_READ_ONLY, actualSizeOfMatrix);
    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, actualSizeOfMatrix);

    cl::Kernel kernel(program, "matrixMultiplication");
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, SIZE);

    cl::CommandQueue queue(context, *device, CL_QUEUE_PROFILING_ENABLE);

    queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, actualSizeOfMatrix, A);
    queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, actualSizeOfMatrix, B);

    cl::NDRange globalWorkSize(SIZE, SIZE);
    cl::NDRange localWorkSize(16, 16); // factor of SIZE

    cl::Event _event;

    auto start = chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, localWorkSize, NULL, &_event);
    _event.wait();

    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, actualSizeOfMatrix, C);
    queue.finish();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> openclTime = end - start;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (C[i * SIZE + j] != localC[i * SIZE + j]) {
                cerr << "Error found! at index [" << i << "," << j << "]" << endl;
                cerr << "Expected Value: " << localC[i * SIZE + j] << " Value Found: " << C[i * SIZE + j] << endl;
                exit(1);
            }
        }
    }

    delete[] C;
    return openclTime.count();
}

int main() {
    srand(time(0));
    int* A, * B, * localC;
    A = new int[SIZE * SIZE];
    B = new int[SIZE * SIZE];
    localC = new int[SIZE * SIZE] {0};
    cout << "Initializing Matrices" << endl;
    initializeMatrix(A);
    initializeMatrix(B);
    cout << "Matrices Initialized" << endl << endl;

    auto start = chrono::high_resolution_clock::now();
    matrixMultiplication(A, B, localC);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> serialTime = end - start;

    vector<cl::Device> allDevices = getAllDevices();
    vector<double> parallelTimes;

    parallelTimes.push_back(serialTime.count());

    for (auto& d : allDevices) {
	    cout << "Running on device: " << d.getInfo<CL_DEVICE_NAME>() << endl;
		double time = runParallel(&d, A, B, localC);
		parallelTimes.push_back(time);
	}

    ofstream file("results.csv", ios::app);
    file << SIZE << ",";
    for (int i = 0; i < parallelTimes.size(); i++) {
        if (i == parallelTimes.size() - 1) {
			file << fixed << parallelTimes[i] << endl;
			break;
		}
		file << fixed << parallelTimes[i] << ",";
	}

    delete[] A;
    delete[] B;
    delete[] localC;

    file.close();
	return 0;
}
*/