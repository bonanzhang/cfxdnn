# - Check for the presence of MKL Runtime library
# The module defines the following variables
# MKL_RT_FOUND        - true if was MKL_RT found on the system
# MKL_RT_INCLUDE_DIRS - Location of the mkl headers
# MKL_RT_LIBRARIES    - Required Libraries for all requested bindings
set(MKL_ROOT "$ENV{MKLROOT}")
find_path(MKL_RT_INCLUDE_DIRS
          mkl_dnn.h
          ${MKL_ROOT}/include
          NO_DEFAULT_PATH)
if (MKL_RT_INCLUDE_DIRS)
else (MKL_RT_INCLUDE_DIRS)
    unset(MKL_RT_FOUND)
    message(FATAL_ERROR "Cannot find MKL Runtime headers!")
endif (MKL_RT_INCLUDE_DIRS)

# MKL_RT_LIBRARIES should include the following on Linux:
# MKL_ROOT/lib/intel64/
# libmkl_rt.so
find_library(MKL_RT_LIBRARY
             mkl_rt
             ${MKL_ROOT}/lib/intel64
             NO_DEFAULT_PATH)
if (MKL_RT_LIBRARY)
    list(APPEND MKL_RT_LIBRARIES ${MKL_RT_LIBRARY})
else (MKL_RT_LIBRARY)
    unset(MKL_RT_FOUND)
    message(FATAL_ERROR "Cannot Find MKL Runtime Library")
endif (MKL_RT_LIBRARY)
set(MKL_RT_FOUND)
