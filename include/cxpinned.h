// This memory allocator is based on the example from
//  https://learn.microsoft.com/en-us/cpp/standard-library/allocators?view=msvc-170
// We have simply replaced malloc and free with cudaMallocHost and cudaFreeHost
//
// Projects using this include file may neded to be compiled as cuda projets in
// order to for the linker to find the cuda functions.  
//
// Copyright Richard Ansorge 2020. 
// Part of cx utility suite devevoped for CUP Publication:
//         Programming in Parallel with CUDA 
//
// cx stands for CUDA Examples, these are bits of code used by the examples
// Comments are welcome please feel free to send emails to me.
// Richard Ansorge, rea1@cam.ac.uk
// version 2.50 March 2023 


#pragma once
#include <stdlib.h>        //size_t
#include "cuda_runtime.h"  // provides cudaMallocHost and cudaFreeHost

namespace cx {
template <class T> struct Pallocator
{
    typedef T value_type;
    Pallocator() noexcept {} //default ctor not required by C++ Standard Library

    // A converting copy constructor:
    template<class U> Pallocator(const Pallocator<U>&) noexcept {}
    template<class U> bool operator==(const Pallocator<U>&) const noexcept
    {
        return true;
    }
    template<class U> bool operator!=(const Pallocator<U>&) const noexcept
    {
        return false;
    }
    T* allocate(const size_t n) const;
    void deallocate(T* const p, size_t) const noexcept;
};

template <class T> T* Pallocator<T>::allocate(const size_t n) const
{
    if (n == 0)
    {
        return nullptr;
    }
    if (n > static_cast<size_t>(-1) / sizeof(T))
    {
        throw std::bad_array_new_length();
    }

    void* pv;  //NB have to drop const prefix here
    cudaMallocHost(&pv, n * sizeof(T));
    if (!pv) { throw std::bad_alloc(); }
    return static_cast<T*>(pv);
}

template<class T> void Pallocator<T>::deallocate(T * const p, size_t) const noexcept
{
    cudaFreeHost(p);
}
} // end namespace cx
