 #ifndef TENSOR_H
 #define TENSOR_H
 
 #include <stdint.h>
 #include <stdbool.h>
 typedef enum {
     TENSOR_FLOAT32,
     TENSOR_FLOAT64,
     TENSOR_INT32,
     TENSOR_INT64,
     TENSOR_UINT8
 } TensorDataType;
 
 typedef struct {
     int64_t* dims;      
     int64_t* strides;   
     int ndim;            
     int64_t size;        
 } TensorShape;
 
 struct Node;
 
 typedef struct Tensor {
     void* data;                   
     TensorShape shape;            
     TensorDataType dtype;         
     bool requires_grad;           
     void* grad;                   
     struct Node* grad_fn;         
     int ref_count;                
     bool owns_data;               
 } Tensor;
 
 
 Tensor* tensor_new(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad);
 
 Tensor* tensor_from_data(void* data, const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad);
 
 Tensor* tensor_zeros(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad);

 Tensor* tensor_ones(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad);
 
 Tensor* tensor_rand(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad);
 
 void tensor_incref(Tensor* tensor);

 void tensor_decref(Tensor* tensor);
 
 void tensor_free(Tensor* tensor);
 
 size_t tensor_dtype_size(TensorDataType dtype);
 

 void tensor_print(const Tensor* tensor);
 
 size_t tensor_bytes(const Tensor* tensor);
 
 #endif 