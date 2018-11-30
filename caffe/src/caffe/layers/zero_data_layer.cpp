#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void ZeroDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	vector<int> shape;
	for(int i = 0; i < this->layer_param_.zero_data_param().dim_size(); ++i) {
		shape.push_back(this->layer_param_.zero_data_param().dim(i));
	}
	top[0]->Reshape(shape);
	caffe_set(top[0]->count(), Dtype(0.5), top[0]->mutable_cpu_data());
}

INSTANTIATE_CLASS(ZeroDataLayer);
REGISTER_LAYER_CLASS(ZeroData);
}
