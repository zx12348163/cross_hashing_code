#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void RegulationLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
  }
  template <typename Dtype>
  void RegulationLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    memset(diff, Dtype(0), bottom[0]->count()*sizeof(Dtype));
    const int num = bottom[0]->num();
    const int dim = bottom[0]->channels();
    Dtype loss = 0;
    for (int i = 0; i < num; ++ i) {
      for (int j = 0; j < dim; ++ j) {
	const Dtype x = bottom_data[i * dim + j];
	if (x >= 0) {
	  loss += Dtype(1) - x;
	  diff[i * dim + j] = 2 * (x - Dtype(1));
	}
	else {
	  loss += Dtype(1) + x;
	  diff[i * dim + j] = 2 * (x + Dtype(1));
	}
      }
    }
    top[0]->mutable_cpu_data()[0] = loss / num;
    caffe_scal(bottom[0]->count(), top[0]->cpu_diff()[0] / num, diff);
  }
  template <typename Dtype>
  void RegulationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  }
  #ifdef CPU_ONLY
    STUB_GPU(RegulationLossLayer);
  #endif

  INSTANTIATE_CLASS(RegulationLossLayer);
  REGISTER_LAYER_CLASS(RegulationLoss);

}
