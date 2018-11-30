#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LstmDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	vector<int> shape;
	T_ = this->layer_param_.lstm_data_param().sq_length();
	N_ = this->layer_param_.lstm_data_param().batch_size();
	//shape.push_back(T_);
	//shape.push_back(N_);
	//top[1]->Reshape(shape);
	//shape.push_back(1);
	//top[0]->Reshape(shape);
	shape.push_back(T_);
	shape.push_back(N_);
	top[0]->Reshape(shape);
}

template <typename Dtype>
void LstmDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
	caffe_set(top[0]->count(), Dtype(1), top[0]->mutable_cpu_data());
	caffe_set(N_, Dtype(0), top[0]->mutable_cpu_data());
}

INSTANTIATE_CLASS(LstmDataLayer);
REGISTER_LAYER_CLASS(LstmData);
}
