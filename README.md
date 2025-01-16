Tutorial:
https://learn.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=msvc-170

get standalone intel DPC++ compiler at
	https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html

installer should offer you the option to integrate the installation into VisualStudio2022, check that box to choose that option. This will ensure that "intel DPC++ compiler 2025" appears as a possible platform toolset in the DLL project's Project Properties.

1. create a DLL C++ project from the VS templates
	DLL .h file format:
	
	#pragma once
	
	#ifdef MathLibrary_EXPORTS
	#define MathLibrary_API __declspec(dllexport)
	#else
	#define MathLibrary_API __declspec(dllimport)
	#endif

	extern "C" MathLibrary_API void func(); // "C", as well as MathLibrary_API, are both necessary. idk why

2.
	DLL .cpp file (NOT dllmain.cpp, a new file) format:
	#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
	// ... other includes as needed
	#include "MathLibrary.h"

	// global variables allowed

	// define all functions
	void func(){
		// function operations
	}
3.
	then build the DLL project


4. create a console application project (preferably in the same solution as the DLL project, so they share a solution and .sln file)
	Client: add additional library directories using $(IntDir), additional include directories, additional dependencies
	*additional dependencies are not the whole path, just the file (ie. "libname.lib" (no quotes though))

	".dll is not a valid win32 application" or something -> set client as startup project in solution explorer
	".lib cannot be found" -> 
	*right click client project, click "build dependencies", "project dependencies", list dll project as a client dependency
	
5.
	if dll project and client project are in different solutions, must put .dll file in client output directory, manually or as a post-build event
	
	
PS: just to be sure, always REbuild the dll project, then build the client project. Otherwise, let's say you update the dll project header file. If you build the client project, then if a .lib and .dll file already exist, then EVEN THOUGH the client project is listed as dependent on the dll project, the .lib and .dll files will not be rebuilt unless they aren't there. This is an issue because the .dll and .lib files don't update, so be safe and just BUILD BOTH PROJECTS EVERY TIME.


LNK1104	cannot open DLL project's .lib file -> Client program properties VC++ directories must include the lib directory for the lib file that the DLL project produces (in "Library Directories" of course)

Had to edit PATH variable on computer in advanced system settings to include the path to the SYCL windows compiler binary, which has all the sycl dlls
	System Properties -> Environment Variables -> User variables for _username_
 	add this path:
  		C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin




SYCL can't handle recursion or non-const global variables. It can when that stuff is run on host and not GPU, but GPU can't handle those things. The more you know...


constant initialized means using constexpr I think...

Git -> reset -> delete changes to bring your local project back to the most recent commit.

SYCL kernel DOES NOT work with doubles. Any time you use 2.0 in a calculation, it must be written 2.0f, which evaluates to a float. If you just write 2.0, it evaluates to a double and the kernel just explodes and lags out. What the fuck man.
