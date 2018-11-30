#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinaryLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
}

template <typename Dtype>
void BinaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
     return;
  }
  caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}


INSTANTIATE_LAYER_GPU_FUNCS(BinaryLayer);

}  // namespace caffe
