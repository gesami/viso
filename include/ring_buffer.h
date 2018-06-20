//
// Created by sage on 09.06.18.
//

#ifndef VISO_RING_BUFFER_H
#define VISO_RING_BUFFER_H

#include <vector>

template <int N, typename T>
class ring_buffer {
public:
    ring_buffer()
        : end_(0)
        , size_(0)
    {
        ring_.resize(N);
    }

    ring_buffer(ring_buffer& rhs) = delete;
    ring_buffer(const ring_buffer& rhs) = delete;
    ring_buffer(volatile ring_buffer& rhs) = delete;
    ring_buffer(const volatile ring_buffer& rhs) = delete;
    ring_buffer(ring_buffer&& rhs) = delete;

    inline void push(T t)
    {
        ring_[end_] = t;
        end_ = (end_ + 1) % N;
        size_ = std::min(size_ + 1, N);
    }

    inline int size() { return size_; }
    inline T last()
    {
        assert(size() > 0);
        return this->operator[](size() - 1);
    }

    inline T operator[](int index)
    {
        int start = end_ - size_;
        if (start < 0) {
            start += N;
        }
        return ring_[(start + index) % N];
    }

    std::vector<T> to_vector()
    {
        std::vector<T> vec(size());
        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = this->operator[](i);
        }

        return vec;
    }

private:
    std::vector<T> ring_;
    int end_;
    int size_;
};

#endif //VISO_RING_BUFFER_H
