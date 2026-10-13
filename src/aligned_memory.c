#include "aligned_memory.h"
#include <stdlib.h>
#include <stdio.h>

// On standard POSIX systems, posix_memalign is available.
// If compiling with C11 or newer without POSIX source defined, we can declare it.
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200112L
extern int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif

void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Error: Failed to allocate aligned memory\n");
        return NULL;
    }
    return ptr;
}

void aligned_free(void* ptr) {
    free(ptr);
}
