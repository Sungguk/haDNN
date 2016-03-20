# $File: Makefile
OBJ_DIR = obj

OPTFLAGS ?= -O3 -march=native
#OPTFLAGS ?= -g -O0
DEFINES ?= -DDEBUG  	# comment out this line to improve speed

INCLUDE_DIR = -I.
LDFLAGS += -lHalide

PKGS = opencv
INCLUDE_DIR += $(shell pkg-config --cflags $(PKGS))
LDFLAGS += $(shell pkg-config $(PKGS) --libs)

CXXFLAGS = $(INCLUDE_DIR) -fopenmp -Wall -Wextra
CXXFLAGS += $(DEFINES) -std=c++14 $(OPTFLAGS)

LDFLAGS += $(OPTFLAGS) -fopenmp

SHELL = bash
ccSOURCES = $(shell find . -name "*.cc" | cut -c3-)
OBJS = $(addprefix $(OBJ_DIR)/,$(ccSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

cppSOURCES = $(shell find . -name "*.cpp" | cut -c3-)
cppOBJS = $(addprefix $(OBJ_DIR)/,$(cppSOURCES:.cpp=.o))
cppDEPFILES = $(cppOBJS:.o=.d)
TARGET = $(cppSOURCES:.cpp=.bin)


.PHONY: all clean run

all: $(TARGET)

ifneq ($(MAKECMDGOALS), clean)
sinclude $(DEPFILES)
sinclude $(cppDEPFILES)
endif

$(OBJS): $(OBJ_DIR)/%.o: %.cc
	@echo "[cc] $< ..."
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

$(cppOBJS): $(OBJ_DIR)/%.o: %.cpp
	@echo "[cc] $< ..."
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

$(DEPFILES): $(OBJ_DIR)/%.d: %.cc  Makefile
	@mkdir -p $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(CXXFLAGS) $(DEFINES) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<"  > "$@"

$(cppDEPFILES): $(OBJ_DIR)/%.d: %.cpp  Makefile
	@mkdir -p $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(CXXFLAGS) $(DEFINES) -MM -MT "$(OBJ_DIR)/$(<:.cpp=.o) $(OBJ_DIR)/$(<:.cpp=.d)" "$<"  > "$@"

$(TARGET): %.bin: $(OBJ_DIR)/%.o $(OBJS)
	@echo "Linking $@ ..."
	@$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)
	@echo Good day!😃

clean:
	@rm -rvf $(OBJ_DIR) $(TARGET)


run: $(TARGET)
	./main.bin
