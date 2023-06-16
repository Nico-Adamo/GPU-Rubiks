# CS 179 Labs 5-6 Makefile
# Written by Aadyot Bhatnagar, 2018

# Input Names
CUDA_FILES = search_gpu.cu
CPP_FILES = search.cpp rubiks.cpp heuristic.cpp
CPP_MAIN = main.cpp

# Directory names
SRCDIR = src
OBJDIR = build
BINDIR = bin

# ------------------------------------------------------------------------------

# CUDA path, compiler, and flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCC_FLAGS := -m32
else
	NVCC_FLAGS := -m64
endif

NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
	      --expt-relaxed-constexpr -D THRUST_IGNORE_DEPRECATED_CPP_DIALECT
NVCC_INCLUDE =
NVCC_CUDA_LIBS =
NVCC_GENCODES = -gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread
INCLUDE = -I$(CUDA_INC_PATH)
CUDA_LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -lcudnn -lcurand


# ------------------------------------------------------------------------------
# Object files
# ------------------------------------------------------------------------------

# CUDA Object Files
CUDA_OBJ = $(OBJDIR)/cuda.o
CUDA_OBJ_FILES = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CUDA_FILES)))

# C++ Object Files
CPP_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CPP_FILES)))
CONV_OBJ = $(addprefix $(OBJDIR)/conv-, $(addsuffix .o, $(CPP_MAIN)))
DENSE_OBJ = $(addprefix $(OBJDIR)/dense-, $(addsuffix .o, $(CPP_MAIN)))

# List of all common objects needed to be linked into the final executable
COMMON_OBJ = $(CPP_OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES)


# ------------------------------------------------------------------------------
# Make rules
# ------------------------------------------------------------------------------

# Top level rules
all: rubik

rubik: $(DENSE_OBJ) $(COMMON_OBJ)
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(CUDA_LIBS)

# Compile C++ Source Files
$(DENSE_OBJ): $(addprefix $(SRCDIR)/, $(CPP_MAIN))
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

$(CPP_OBJ): $(OBJDIR)/%.o : $(SRCDIR)/%
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<


# Compile CUDA Source Files
$(CUDA_OBJ_FILES): $(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $<


# Clean everything including temporary Emacs files
clean:
	rm -f $(BINDIR)/* $(OBJDIR)/*.o $(SRCDIR)/*~ *~


.PHONY: clean all



# # Compiler and flags
# CXX = nvcc
# CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# # Directories
# SRC_DIR = src
# BUILD_DIR = build
# BIN_DIR = bin

# # Source files
# SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
# TARGET = $(BIN_DIR)/rubik

# # Build target
# $(TARGET): $(OBJS)
# 	@mkdir -p $(BIN_DIR)
# 	$(CXX) $(CXXFLAGS) $^ -o $@

# # Compile source files
# $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
# 	@mkdir -p $(BUILD_DIR)
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# # Clean build artifacts
# clean:
# 	rm -rf $(BUILD_DIR) $(BIN_DIR)

# # Phony targets
# .PHONY: all clean

# all: $(TARGET)
