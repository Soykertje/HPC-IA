# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug

# Include any dependencies generated for this target.
include CMakeFiles/RegresionLineal.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RegresionLineal.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RegresionLineal.dir/flags.make

CMakeFiles/RegresionLineal.dir/main.cpp.o: CMakeFiles/RegresionLineal.dir/flags.make
CMakeFiles/RegresionLineal.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RegresionLineal.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RegresionLineal.dir/main.cpp.o -c /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/main.cpp

CMakeFiles/RegresionLineal.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RegresionLineal.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/main.cpp > CMakeFiles/RegresionLineal.dir/main.cpp.i

CMakeFiles/RegresionLineal.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RegresionLineal.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/main.cpp -o CMakeFiles/RegresionLineal.dir/main.cpp.s

CMakeFiles/RegresionLineal.dir/linearregresion.cpp.o: CMakeFiles/RegresionLineal.dir/flags.make
CMakeFiles/RegresionLineal.dir/linearregresion.cpp.o: ../linearregresion.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/RegresionLineal.dir/linearregresion.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RegresionLineal.dir/linearregresion.cpp.o -c /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/linearregresion.cpp

CMakeFiles/RegresionLineal.dir/linearregresion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RegresionLineal.dir/linearregresion.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/linearregresion.cpp > CMakeFiles/RegresionLineal.dir/linearregresion.cpp.i

CMakeFiles/RegresionLineal.dir/linearregresion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RegresionLineal.dir/linearregresion.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/linearregresion.cpp -o CMakeFiles/RegresionLineal.dir/linearregresion.cpp.s

CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.o: CMakeFiles/RegresionLineal.dir/flags.make
CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.o: ../extraerdata/extraerdata.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.o -c /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/extraerdata/extraerdata.cpp

CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/extraerdata/extraerdata.cpp > CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.i

CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/extraerdata/extraerdata.cpp -o CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.s

# Object files for target RegresionLineal
RegresionLineal_OBJECTS = \
"CMakeFiles/RegresionLineal.dir/main.cpp.o" \
"CMakeFiles/RegresionLineal.dir/linearregresion.cpp.o" \
"CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.o"

# External object files for target RegresionLineal
RegresionLineal_EXTERNAL_OBJECTS =

RegresionLineal: CMakeFiles/RegresionLineal.dir/main.cpp.o
RegresionLineal: CMakeFiles/RegresionLineal.dir/linearregresion.cpp.o
RegresionLineal: CMakeFiles/RegresionLineal.dir/extraerdata/extraerdata.cpp.o
RegresionLineal: CMakeFiles/RegresionLineal.dir/build.make
RegresionLineal: CMakeFiles/RegresionLineal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable RegresionLineal"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RegresionLineal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RegresionLineal.dir/build: RegresionLineal

.PHONY : CMakeFiles/RegresionLineal.dir/build

CMakeFiles/RegresionLineal.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RegresionLineal.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RegresionLineal.dir/clean

CMakeFiles/RegresionLineal.dir/depend:
	cd /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug /home/andres/Documents/NOVENO_SEMESTRE/HPC/RegresionLineal/Debug/CMakeFiles/RegresionLineal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RegresionLineal.dir/depend

