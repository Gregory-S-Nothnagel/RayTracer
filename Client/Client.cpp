#define _USE_MATH_DEFINES // for M_PI
#define NOMINMAX // so that std::min is used instead of the windows.h version
#include <windows.h> // Includes the core Windows API header file, which provides functions for creating windows, handling events, and drawing on the screen.
#include <windowsx.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include "RayTracerSYCL.h"

// Global Variables
std::map<int, bool> key_pressed = { {'W', false}, {'A', false}, {'S', false}, {'D', false} }; // which buttons are being pressed
int frames_still = 0;
double eye_pos[3] = { 0, 0, 0 };
double eye_rotation[2] = { 0, 0 }; // x and y, radians (default view direction is (0, 0, 1))
const int WIDTH = 256, HEIGHT = 256;
unsigned char image_data[WIDTH * HEIGHT * 3];
double image_data_double[WIDTH * HEIGHT * 3];
uint64_t rand_seeds[WIDTH * HEIGHT]; // initialized in WinMain

// Global Functions

// faster pow() function for when using positive integer powers
double power(double x, int p) {
	double product = 1;
	for (int i = 0; i < p; i++) product *= x;
	return product;
}
// simple normalization function
void normalize(double* v, int dim) {
	double length = 0;
	for (int i = 0; i < dim; i++) {
		length += power(v[i], 2);
	}
	length = sqrt(length);
	for (int i = 0; i < dim; i++) {
		v[i] /= length;
	}
}
// rotates 3d vector around x-axis
void rotateX(double* v, double angle) { // angle in radians
	double temp[3] = { v[0], cos(angle) * v[1] - sin(angle) * v[2], sin(angle) * v[1] + cos(angle) * v[2] };
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
// rotates 3d vector around y-axis
void rotateY(double* v, double angle) { // angle in radians
	double temp[3] = {
		cos(angle) * v[0] + sin(angle) * v[2],
		v[1],
		-sin(angle) * v[0] + cos(angle) * v[2]
	};
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}

// Window Procedure function - handles messages sent to the window (e.g., paint, close, keypress, timers, etc.)
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {

	// key press events
	if (uMsg == WM_KEYDOWN) {
		key_pressed[wParam] = true;
	}
	if (uMsg == WM_KEYUP) {
		key_pressed[wParam] = false;
	}

	// mouse move events
	if (uMsg == WM_MOUSEMOVE) {

		frames_still = 0;

		// get mouse coords
		int x_pos = GET_X_LPARAM(lParam);
		int y_pos = GET_Y_LPARAM(lParam);

		eye_rotation[1] = M_PI * (2 * double(x_pos) / WIDTH - 1);
		eye_rotation[0] = -M_PI * (double(y_pos) / WIDTH - .5);

	}

	// If the message is "WM_PAINT", the window needs to be redrawn.
	if (uMsg == WM_PAINT) {
		PAINTSTRUCT ps;
		// Begin the paint operation and get a handle to the device context (HDC) for drawing.
		HDC hdc = BeginPaint(hwnd, &ps);

		// Initialize a BITMAPINFO structure to describe the image format.
		BITMAPINFO bmi = {};
		bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader); // Set the size of the BITMAPINFOHEADER.
		bmi.bmiHeader.biWidth = WIDTH;               // Set the width of the image in pixels.
		bmi.bmiHeader.biHeight = -HEIGHT;            // Set the height of the image. Negative for top-down orientation.
		bmi.bmiHeader.biPlanes = 1;                  // Always set to 1 (required by Windows API).
		bmi.bmiHeader.biBitCount = 24;               // Specifies 24 bits per pixel (RGB format).
		bmi.bmiHeader.biCompression = BI_RGB;        // Specifies no compression (raw RGB data).

		// Draw the image data onto the window using StretchDIBits
		StretchDIBits(
			hdc,                          // Handle to the device context for the window.
			0, 0, WIDTH, HEIGHT,          // Destination rectangle on the screen.
			0, 0, WIDTH, HEIGHT,          // Source rectangle from the image.
			image_data,                    // Pointer to the raw image data.
			&bmi,                         // Pointer to the BITMAPINFO structure describing the image.
			DIB_RGB_COLORS,               // Indicates that the image uses RGB colors (not a palette).
			SRCCOPY                       // Raster operation code - simply copy the source pixels to the destination.
		);

		// End the paint operation and release resources.
		EndPaint(hwnd, &ps);
		return 0;
	}

	// This block is triggered every time the timer fires
	if (uMsg == WM_TIMER) {

		// change eye position based on key presses and view direction
		if (key_pressed['W'] || key_pressed['A'] || key_pressed['S'] || key_pressed['D']) {

			double move_dir[3] = { 0, 0, 0 };
			if (key_pressed['W']) move_dir[2] += 1;
			if (key_pressed['A']) move_dir[0] -= 1;
			if (key_pressed['S']) move_dir[2] -= 1;
			if (key_pressed['D']) move_dir[0] += 1;
			if (!(key_pressed['W'] && key_pressed['S']) && !(key_pressed['A'] && key_pressed['D'])) normalize(move_dir, 3);

			rotateY(move_dir, eye_rotation[1]);

			eye_pos[0] += move_dir[0] * .01;
			eye_pos[2] += move_dir[2] * .01;

		}

		func(WIDTH, HEIGHT, image_data, image_data_double, frames_still, eye_rotation, eye_pos, rand_seeds);

		frames_still++;

		// trigger WM_PAINT to redraw
		InvalidateRect(hwnd, nullptr, TRUE);
		return 0; // Don't pass it to default procedure
	}

	// if window close message received
	if (uMsg == WM_DESTROY) {
		KillTimer(hwnd, 1); // Stop the timer when the window is closed
		PostQuitMessage(0); // Exit the message loop
		return 0;
	}

	// For all other messages, use the default window procedure provided by Windows.
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// The entry point for a Windows application. This is where the program starts.
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {

	// initializing rand_seeds for all pixels
	for (int i = 0; i < WIDTH * HEIGHT; i++) rand_seeds[i] = i;

	// Define the properties of the window class.
	WNDCLASS wc = { 0 };                   // Zero-initialize the WNDCLASS structure.
	wc.lpfnWndProc = WindowProc;           // Set the callback function that handles this window's messages.
	wc.hInstance = hInstance;              // Handle to the current instance of the application.
	wc.lpszClassName = L"MinimalImage";     // Name of the window class.

	// Register the window class with the operating system.
	RegisterClass(&wc);

	// Initialize the window using the registered window class.
	HWND hwnd = CreateWindow(
		wc.lpszClassName,                 // Name of the registered window class.
		L"Image Viewer",                   // Title of the window.
		WS_OVERLAPPEDWINDOW,              // Style of the window (includes borders, title bar, etc.).
		CW_USEDEFAULT, CW_USEDEFAULT,     // Position of the window (default values).
		WIDTH, HEIGHT,                    // Width and height of the window.
		nullptr,                          // No parent window (top-level window).
		nullptr,                          // No menu for this window.
		hInstance,                        // Handle to the application instance.
		nullptr                           // No additional application data.
	);

	// Make the window visible on the screen.
	ShowWindow(hwnd, nCmdShow);

	// Set a timer with ID 1 that fires every x milliseconds (e.g., 1000ms = 1 second)
	SetTimer(hwnd, 1, 25, NULL); // 1000ms = 1 second


	// Main event loop - processes messages sent to the application (e.g., mouse clicks, key presses)
	MSG msg;
	while (GetMessage(&msg, nullptr, 0, 0)) { // Retrieve messages from the queue
		TranslateMessage(&msg); // Translate virtual-key messages into character messages
		DispatchMessage(&msg);  // Dispatch the message to the appropriate window procedure
	}

	// Exit the application when the message loop ends.
	return 0;
}
