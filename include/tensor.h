#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stdint.h>
typedef enum { //supported datatypes
    TENSOR_FLOAT32,
    TENSOR_FLOAT64,
    TENSOR_INT32,
    TENSOR_INT64,
    TENSOR_UINT8
}TensorDataType;

typedef struct {
    int64_t* dims;
    int64_t* strides;
    int ndim;
    int64_t size;
}TensorShape;

struct Node;

typedef struct tensor //main tensor struct
{
    void* data;
    TensorShape* shape;
    TensorDataType* type;
    bool owns_data; //whter own data or just a view
}Tensor;

Tensor* tensor_new(const int64_t* dims, int ndim, TensorDataType dtype);

Tensor* tensor_from_data(const int64_t* dims, int ndim, TensorDataType dtype);

Tensor* tensor_ones(const int64_t* dims, int ndim, TensorDataType dtype);

Tensor* tensor_rand(const int64_t* dims, int ndim, TensorDataType dtype);

void Tensor_free(Tensor* tensor);

size_t tensor_dtype_size(TensorDataType dtype);

void tensor_print(const Tensor* tensor);

size_t tensor_bytes(const Tensor* tensor);
#endif /*TENSOR_H*/