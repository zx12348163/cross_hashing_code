#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LstmLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	T_ = this->layer_param_.lstm_label_param().sq_length();
	vector<int> shape;
	shape.push_back(bottom[0]->shape(0));
	shape.push_back(T_);
	top[0]->Reshape(shape);
	//top[1]->Reshape(shape);
}
template <typename Dtype>
void LstmLabelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//caffe_set(top[1]->count(), Dtype(-1), top[1]->mutable_cpu_data());
	const Dtype* bottom_data = bottom[0]->cpu_data();
	//Dtype* maker_data = top[0]->mutable_cpu_data();
	Dtype* label_data = top[0]->mutable_cpu_data();
	const int num = bottom[0]->num();
	const int dim = bottom[0]->count()/num;
	caffe_set(top[0]->count(), Dtype(dim),top[0]->mutable_cpu_data());
	for(int i = 0; i < num; ++i) {
		int label_ind = 0;
		//int maker_ind = 1;
		for(int j = 0; j < dim; ++j) {
			if(1 == bottom_data[i*dim+j]) {
				label_data[i * T_ + label_ind] = j;
				//maker_data[i+maker_ind*num] = 1;
				label_ind++;
				//maker_ind++;
			}
		}
		label_data[i * T_ + label_ind] = dim;
	}
}

INSTANTIATE_CLASS(LstmLabelLayer);
REGISTER_LAYER_CLASS(LstmLabel);
}
