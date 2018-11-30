#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime> 
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	alpha_ = this->layer_param_.weight_loss_param().alpha();
}

template <typename Dtype>
void WeightLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void WeightLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count()/num;
	Dtype loss = 0;
	for(int i = 0; i < num; ++i) {
		Dtype sum = 0;
		for(int k = 0; k < dim; ++k) {
			sum += bottom_data[i*dim+k];
		}
		loss += pow((alpha_*dim - sum),2);
	}
	loss = loss / Dtype(num*dim) / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count() / num;
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	for(int i = 0; i < num; ++i) {
		Dtype sum = 0;
		for(int k = 0; k < dim; ++k) {
			sum += bottom_data[i*dim+k];
		}
		caffe_set(dim, -sum / Dtype(num*dim) * loss_weight, bottom_diff+i*dim);
	}
}

#ifdef CPU_ONLY
STUB_GPU(WeightLossLayer);
#endif

INSTANTIATE_CLASS(WeightLossLayer);
REGISTER_LAYER_CLASS(WeightLoss);

}  // namespace caffe





