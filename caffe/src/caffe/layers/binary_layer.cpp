#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinaryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(count_, top[0]->count());
}

template <typename Dtype>
void BinaryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int i = 0; i < num; ++ i) {
    for (int j = 0; j < channels * height * width; ++ j) {
      if(bottom_data[j] > (Dtype)1.0 / (Dtype)(channels*height*width)) {
	top_data[j] = Dtype(0);
      }
      else top_data[j] = Dtype(1);
    }
    /**********
    const int SELECT = 7;
    vector<pair<Dtype, int> > v;
    for (int j = 0; j < channels * height * width; ++ j) {
      v.push_back(make_pair(bottom_data[j], j));
      top_data[j] = Dtype(0);
    }
    sort(v.begin(), v.end());
    for (int j = 0; j < SELECT; ++ j) {
      top_data[v[v.size()-1-j].second] = 1;
    }
    /********/
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
  }
  /***
  const int length = std::sqrt(channels);
  std::cout << "a=np.array([";
  for(int i = 0; i < num; ++ i) {
    if(i > 0) std::cout << ",";
    std::cout << "[";
    for (int j = 0; j < length; ++ j) {
      std::cout<< "[";
      for (int k = 0; k < length; ++ k) {
	if (k > 0) std::cout << ",";
	std::cout << top[0]->data_at(i, j * length + k, 0, 0);
      }
      std::cout << "]";
      if (j < length - 1) std::cout << ",";
    }
    std::cout << "]";
  }
  std::cout << "])" << std::endl;
  /****/

}

template <typename Dtype>
void BinaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());

}


#ifdef CPU_ONLY
STUB_GPU(BinaryLayer);
#endif

INSTANTIATE_CLASS(BinaryLayer);
REGISTER_LAYER_CLASS(Binary);

}  // namespace caffe
