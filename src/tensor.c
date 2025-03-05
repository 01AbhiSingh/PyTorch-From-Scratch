
 #include "../include/tensor.h"
 #include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 #include <assert.h>
 
 
 static void calculate_strides(TensorShape* shape) {
     if (shape->ndim <= 0) return;
     
     shape->strides = (int64_t*)malloc(shape->ndim * sizeof(int64_t));
     if (!shape->strides) {
         fprintf(stderr, "Failed to allocate memory for strides\n");
         exit(EXIT_FAILURE);
     }
     
     shape->strides[shape->ndim - 1] = 1;
     for (int i = shape->ndim - 2; i >= 0; i--) {
         shape->strides[i] = shape->strides[i + 1] * shape->dims[i + 1];
     }
 }
 
 static int64_t calculate_size(const int64_t* dims, int ndim) {
     if (ndim <= 0) return 0;
     
     int64_t size = 1;
     for (int i = 0; i < ndim; i++) {
         size *= dims[i];
     }
     return size;
 }
 
 size_t tensor_dtype_size(TensorDataType dtype) {
     switch (dtype) {
         case TENSOR_FLOAT32: return sizeof(float);
         case TENSOR_FLOAT64: return sizeof(double);
         case TENSOR_INT32:   return sizeof(int32_t);
         case TENSOR_INT64:   return sizeof(int64_t);
         case TENSOR_UINT8:   return sizeof(uint8_t);
         default:
             fprintf(stderr, "Unknown tensor data type\n");
             exit(EXIT_FAILURE);
     }
 }
 
 Tensor* tensor_new(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad) {
     if (ndim <= 0 || !dims) {
         fprintf(stderr, "Invalid dimensions for tensor\n");
         return NULL;
     }
     
     Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
     if (!tensor) {
         fprintf(stderr, "Failed to allocate tensor\n");
         return NULL;
     }
     
     tensor->shape.ndim = ndim;
     tensor->shape.dims = (int64_t*)malloc(ndim * sizeof(int64_t));
     if (!tensor->shape.dims) {
         fprintf(stderr, "Failed to allocate tensor dimensions\n");
         free(tensor);
         return NULL;
     }
     
     memcpy(tensor->shape.dims, dims, ndim * sizeof(int64_t));
     
     tensor->shape.size = calculate_size(dims, ndim);
     calculate_strides(&tensor->shape);
     
     size_t bytes = tensor->shape.size * tensor_dtype_size(dtype);
     tensor->data = malloc(bytes);
     if (!tensor->data) {
         fprintf(stderr, "Failed to allocate tensor data (%zu bytes)\n", bytes);
         free(tensor->shape.dims);
         free(tensor->shape.strides);
         free(tensor);
         return NULL;
     }
     
     tensor->dtype = dtype;
     tensor->requires_grad = requires_grad;
     tensor->grad = NULL;
     tensor->grad_fn = NULL;
     tensor->ref_count = 1;
     tensor->owns_data = true;
     
     if (requires_grad) {
         tensor->grad = calloc(tensor->shape.size, tensor_dtype_size(dtype));
         if (!tensor->grad) {
             fprintf(stderr, "Failed to allocate gradient data\n");
             free(tensor->data);
             free(tensor->shape.dims);
             free(tensor->shape.strides);
             free(tensor);
             return NULL;
         }
     }
     
     return tensor;
 }
 
 Tensor* tensor_from_data(void* data, const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad) {
     Tensor* tensor = tensor_new(dims, ndim, dtype, requires_grad);
     if (!tensor) return NULL;
     
     size_t bytes = tensor->shape.size * tensor_dtype_size(dtype);
     memcpy(tensor->data, data, bytes);
     
     return tensor;
 }
 
 Tensor* tensor_zeros(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad) {
     Tensor* tensor = tensor_new(dims, ndim, dtype, requires_grad);
     if (!tensor) return NULL;
     
     size_t bytes = tensor->shape.size * tensor_dtype_size(dtype);
     memset(tensor->data, 0, bytes);
     
     return tensor;
 }
 
 Tensor* tensor_ones(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad) {
     Tensor* tensor = tensor_new(dims, ndim, dtype, requires_grad);
     if (!tensor) return NULL;
     
     size_t elem_size = tensor_dtype_size(dtype);
     
     switch (dtype) {
         case TENSOR_FLOAT32: {
             float* data = (float*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = 1.0f;
             }
             break;
         }
         case TENSOR_FLOAT64: {
             double* data = (double*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = 1.0;
             }
             break;
         }
         case TENSOR_INT32: {
             int32_t* data = (int32_t*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = 1;
             }
             break;
         }
         case TENSOR_INT64: {
             int64_t* data = (int64_t*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = 1;
             }
             break;
         }
         case TENSOR_UINT8: {
             uint8_t* data = (uint8_t*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = 1;
             }
             break;
         }
     }
     
     return tensor;
 }
 
 Tensor* tensor_rand(const int64_t* dims, int ndim, TensorDataType dtype, bool requires_grad) {
     Tensor* tensor = tensor_new(dims, ndim, dtype, requires_grad);
     if (!tensor) return NULL;
     
     switch (dtype) {
         case TENSOR_FLOAT32: {
             float* data = (float*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = (float)rand() / RAND_MAX;
             }
             break;
         }
         case TENSOR_FLOAT64: {
             double* data = (double*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = (double)rand() / RAND_MAX;
             }
             break;
         }
         case TENSOR_INT32: {
             int32_t* data = (int32_t*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = rand() % 100;
             }
             break;
         }
         case TENSOR_INT64: {
             int64_t* data = (int64_t*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = rand() % 100;
             }
             break;
         }
         case TENSOR_UINT8: {
             uint8_t* data = (uint8_t*)tensor->data;
             for (int64_t i = 0; i < tensor->shape.size; i++) {
                 data[i] = rand() % 256;
             }
             break;
         }
     }
     
     return tensor;
 }
 
 void tensor_incref(Tensor* tensor) {
     if (!tensor) return;
     tensor->ref_count++;
 }
 
 void tensor_decref(Tensor* tensor) {
     if (!tensor) return;
     
     tensor->ref_count--;
     if (tensor->ref_count <= 0) {
         tensor_free(tensor);
     }
 }
 
 void tensor_free(Tensor* tensor) {
     if (!tensor) return;
     

     if (tensor->owns_data && tensor->data) {
         free(tensor->data);
     }
     

     if (tensor->grad) {
         free(tensor->grad);
     }
     
     if (tensor->shape.dims) {
         free(tensor->shape.dims);
     }
     
     if (tensor->shape.strides) {
         free(tensor->shape.strides);
     }

     free(tensor);
 }
 
 size_t tensor_bytes(const Tensor* tensor) {
     if (!tensor) return 0;
     return tensor->shape.size * tensor_dtype_size(tensor->dtype);
 }
 
 void tensor_print(const Tensor* tensor) {
     if (!tensor) {
         printf("NULL tensor\n");
         return;
     }
     
     printf("Tensor {\n");
     printf("  dtype: ");
     switch (tensor->dtype) {
         case TENSOR_FLOAT32: printf("float32"); break;
         case TENSOR_FLOAT64: printf("float64"); break;
         case TENSOR_INT32: printf("int32"); break;
         case TENSOR_INT64: printf("int64"); break;
         case TENSOR_UINT8: printf("uint8"); break;
         default: printf("unknown");
     }
     printf("\n");
     
     printf("  shape: [");
     for (int i = 0; i < tensor->shape.ndim; i++) {
         printf("%ld", tensor->shape.dims[i]);
         if (i < tensor->shape.ndim - 1) printf(", ");
     }
     printf("]\n");
     
     printf("  requires_grad: %s\n", tensor->requires_grad ? "true" : "false");
     printf("  size: %ld\n", tensor->shape.size);
     printf("  memory: %zu bytes\n", tensor_bytes(tensor));
     printf("  ref_count: %d\n", tensor->ref_count);
     
     printf("  data: ");
     const int max_elements = 10;
     int elements_to_print = tensor->shape.size < max_elements ? tensor->shape.size : max_elements;
     
     switch (tensor->dtype) {
         case TENSOR_FLOAT32: {
             float* data = (float*)tensor->data;
             printf("[");
             for (int i = 0; i < elements_to_print; i++) {
                 printf("%.4f", data[i]);
                 if (i < elements_to_print - 1) printf(", ");
             }
             if (tensor->shape.size > max_elements) printf(", ...");
             printf("]\n");
             break;
         }
         case TENSOR_FLOAT64: {
             double* data = (double*)tensor->data;
             printf("[");
             for (int i = 0; i < elements_to_print; i++) {
                 printf("%.4f", data[i]);
                 if (i < elements_to_print - 1) printf(", ");
             }
             if (tensor->shape.size > max_elements) printf(", ...");
             printf("]\n");
             break;
         }
         default:
             printf("[...]\n");
     }
     printf("}\n");
 }