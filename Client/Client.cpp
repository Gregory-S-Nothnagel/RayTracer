#define _USE_MATH_DEFINES // for M_PI
#define NOMINMAX // so that std::min is used instead of the windows.h version
#include <windows.h> // Includes the core Windows API header file, which provides functions for creating windows, handling events, and drawing on the screen.
#include <windowsx.h>
#include <iostream>
#include <fstream>
#include <map>
#include "RayTracerSYCL.h"

constexpr auto PI = 3.14159265358979323846f;

// Global Variables
std::map<int, bool> key_pressed = { {'W', false}, {'A', false}, {'S', false}, {'D', false}, {VK_SHIFT, false}, {' ', false} }; // which buttons are being pressed
int frames_still = 0;
float eye_pos[3] = { 0, 0, 0 };
float eye_rotation[2] = { 0, 0 }; // x and y, radians (default view direction is (0, 0, 1))
const int WIDTH = 500, HEIGHT = 500;
unsigned char image_data[WIDTH * HEIGHT * 3];
float image_data_float[WIDTH * HEIGHT * 3];
uint32_t rand_seeds[WIDTH * HEIGHT]; // initialized in WinMain

// Global Functions

// faster pow() function for when using positive integer powers
float power(float x, int p) {
	float product = 1;
	for (int i = 0; i < p; i++) product *= x;
	return product;
}
// simple normalization function
void normalize(float* v, int dim) {
	float length = 0;
	for (int i = 0; i < dim; i++) {
		length += power(v[i], 2);
	}
	length = sqrt(length);
	for (int i = 0; i < dim; i++) {
		v[i] /= length;
	}
}
// rotates 3d vector around x-axis
void rotateX(float* v, float angle) { // angle in radians
	float temp[3] = { v[0], cos(angle) * v[1] - sin(angle) * v[2], sin(angle) * v[1] + cos(angle) * v[2] };
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
// rotates 3d vector around y-axis
void rotateY(float* v, float angle) { // angle in radians
	float temp[3] = {
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
	if (uMsg == WM_KEYDOWN) key_pressed[wParam] = true;
	if (uMsg == WM_KEYUP) key_pressed[wParam] = false;

	// mouse move events
	if (uMsg == WM_MOUSEMOVE) {

		frames_still = 0;

		// get mouse coords
		int x_pos = GET_X_LPARAM(lParam);
		int y_pos = GET_Y_LPARAM(lParam);

		eye_rotation[1] = PI * (2 * float(x_pos) / WIDTH - 1);
		eye_rotation[0] = -PI * (float(y_pos) / WIDTH - .5f);

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
			image_data,                   // Pointer to the raw image data.
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
		if (key_pressed['W'] || key_pressed['A'] || key_pressed['S'] || key_pressed['D'] || key_pressed[VK_SHIFT] || key_pressed[' ']) {

			float move_dir[3] = { 0, 0, 0 };
			if (key_pressed['W']) move_dir[2] += 1;
			if (key_pressed['A']) move_dir[0] -= 1;
			if (key_pressed['S']) move_dir[2] -= 1;
			if (key_pressed['D']) move_dir[0] += 1;
			if (key_pressed[VK_SHIFT]) move_dir[1] += 1;
			if (key_pressed[' ']) move_dir[1] -= 1;
			if (!(key_pressed['W'] && key_pressed['S']) && !(key_pressed['A'] && key_pressed['D']) && !(key_pressed[VK_SHIFT] && key_pressed[' '])) normalize(move_dir, 3);

			rotateY(move_dir, eye_rotation[1]);

			for (int dim = 0; dim < 3; dim++) {
				eye_pos[dim] += move_dir[dim] * .1;
			}
			
		}

		func(WIDTH, HEIGHT, image_data, image_data_float, frames_still, eye_rotation, eye_pos, rand_seeds);

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
