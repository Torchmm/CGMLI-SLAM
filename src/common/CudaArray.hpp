#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace cgmli::cuda {

/// Reusable device or pinned host allocation. The owner synchronizes its
/// stream before changing the capacity or destroying the workspace.
template<class T,bool Pinned=false>
class Array {
public:
    Array()=default;
    ~Array() { release(); }
    Array(const Array&)=delete;
    Array& operator=(const Array&)=delete;
    bool reserve(std::size_t count) {
        if(count<=capacity_) return true;
        void* buffer=nullptr;
        const auto status=Pinned?cudaMallocHost(&buffer,count*sizeof(T)):cudaMalloc(&buffer,count*sizeof(T));
        if(status!=cudaSuccess) return false;
        release();
        pointer_=static_cast<T*>(buffer);
        capacity_=count;
        return true;
    }
    T* data() const { return pointer_; }
    void swap(Array& other) noexcept {
        std::swap(pointer_,other.pointer_);
        std::swap(capacity_,other.capacity_);
    }
private:
    void release() {
        if(pointer_) {
            if constexpr(Pinned) cudaFreeHost(pointer_);
            else cudaFree(pointer_);
        }
    }
    T* pointer_=nullptr;
    std::size_t capacity_=0;
};

}
