
 #include "../include/tensor.h"
 #include <stdio.h>
 #include <stdlib.h>
 #include <assert.h>
 
 #define TEST(name) void test_##name(void)
 #define RUN_TEST(name) do { \
     printf("Running test: %s\n", #name); \
     test_##name(); \
     printf("PASSED: %s\n\n", #name); \
 } while(0)
 
 TEST(tensor_creation) {
  
     int64_t dims[] = {2, 3};
     Tensor* t = tensor_new(dims, 2, TENSOR_FLOAT32, false);
     
     assert(t != NULL);
     assert(t->shape.ndim == 2);
     assert(t->shape.dims[0] == 2);
     assert(t->shape.dims[1] == 3);
     assert(t->shape.size == 6);
     assert(t->dtype == TENSOR_FLOAT32);
     assert(t->requires_grad == false);
     assert(t->grad == NULL);
     assert(t->ref_count == 1);
     

     tensor_free(t);
 }
 
 TEST(tensor_zeros) {
     int64_t dims[] = {2, 2};
     Tensor* t = tensor_zeros(dims, 2, TENSOR_FLOAT32, false);
     
     assert(t != NULL);
     assert(t->shape.size == 4);
     
     float* data = (float*)t->data;
     for (int i = 0; i < t->shape.size; i++) {
         assert(data[i] == 0.0f);
     }
     
     tensor_free(t);
 }
 
 TEST(tensor_ones) {
     int64_t dims[] = {2, 2};
     Tensor* t = tensor_ones(dims, 2, TENSOR_FLOAT32, false);
     
     assert(t != NULL);
     assert(t->shape.size == 4);
     
     float* data = (float*)t->data;
     for (int i = 0; i < t->shape.size; i++) {
         assert(data[i] == 1.0f);
     }
     
     tensor_free(t);
 }
 
 TEST(tensor_from_data) {
     float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
     int64_t dims[] = {2, 2};
     
     Tensor* t = tensor_from_data(data, dims, 2, TENSOR_FLOAT32, false);
     
     assert(t != NULL);
     assert(t->shape.size == 4);
     
     float* t_data = (float*)t->data;
     for (int i = 0; i < t->shape.size; i++) {
         assert(t_data[i] == data[i]);
     }
     
     // Clean up
     tensor_free(t);
 }
 
 TEST(tensor_strides) {
     {
         int64_t dims[] = {2, 3};
         Tensor* t = tensor_new(dims, 2, TENSOR_FLOAT32, false);
         
         assert(t->shape.strides[0] == 3); 
         assert(t->shape.strides[1] == 1); 
         
         tensor_free(t);
     }
     
     {
         int64_t dims[] = {2, 3, 4};
         Tensor* t = tensor_new(dims, 3, TENSOR_FLOAT32, false);
         
         assert(t->shape.strides[0] == 12); 
         assert(t->shape.strides[1] == 4);  
         assert(t->shape.strides[2] == 1);  
         
         tensor_free(t);
     }
 }
 
 TEST(tensor_requires_grad) {
     int64_t dims[] = {2, 2};
     Tensor* t = tensor_zeros(dims, 2, TENSOR_FLOAT32, true);
     
     assert(t != NULL);
     assert(t->requires_grad == true);
     assert(t->grad != NULL);
     
     float* grad_data = (float*)t->grad;
     for (int i = 0; i < t->shape.size; i++) {
         assert(grad_data[i] == 0.0f);
     }
     
     tensor_free(t);
 }
 
 TEST(tensor_ref_counting) {
     int64_t dims[] = {2, 2};
     Tensor* t = tensor_zeros(dims, 2, TENSOR_FLOAT32, false);
     
     assert(t->ref_count == 1);
     
     tensor_incref(t);
     assert(t->ref_count == 2);
     
     tensor_decref(t);
     assert(t->ref_count == 1);
     
     tensor_decref(t);
 }
 
 int main() {
     printf("Running Tensor tests...\n\n");
     
     RUN_TEST(tensor_creation);
     RUN_TEST(tensor_zeros);
     RUN_TEST(tensor_ones);
     RUN_TEST(tensor_from_data);
     RUN_TEST(tensor_strides);
     RUN_TEST(tensor_requires_grad);
     RUN_TEST(tensor_ref_counting);
     
     printf("All tests passed!\n");
     return 0;
 }