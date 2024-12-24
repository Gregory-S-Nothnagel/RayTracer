#pragma once

//extern "C" __declspec(dllexport) void func(int WIDTH, int HEIGHT, unsigned char* image_data, double* image_data_double, int frames_still, double* eye_rotation, double* eye_pos);
extern "C" __declspec(dllexport) void func(int WIDTH, int HEIGHT, unsigned char* image_data, double* image_data_double, int frames_still, double* eye_rotation, double* eye_pos, uint64_t* rand_seeds);
