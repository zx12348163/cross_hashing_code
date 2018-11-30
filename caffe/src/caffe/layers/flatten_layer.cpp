#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int start_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().axis());
  const int end_axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.flatten_param().end_axis());
  vector<int> top_shape;
  for (int i = 0; i < start_axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
  top_shape.push_back(flattened_dim);
  for (int i = end_axis + 1; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void FlattenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
  //const Dtype* m_top = top[0]->cpu_data();
  //for (int i = 0; i < top[0]->num(); ++ i) {
  //  std::cout<<i+1<<std::endl;
  //  for (int j = 0; j < 6; ++ j) {
  //    for (int k = 0; k < 6; ++ k) {
  //	  const Dtype temp = top[0]->data_at(i, j*6+k, 0, 0);
  //	  std::cout << temp << " ";
  //    }
  //    std::cout << std::endl;
  //  }
  //  std::cout << std::endl;
  //}
  //LOG(INFO) << top[0]->num() << top[0]->channels() << top[0]->height() << top[0]->width();
}

template <typename Dtype>
void FlattenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
  //added by ysm 
  //static int print_count = 0;
  //if(print_count%100 == 0) {
  //const Dtype* bottom_diff = bottom[0]->cpu_diff();
  //const Dtype* top_diff = top[0]->cpu_diff();
  //Dtype sum;
  //Dtype max;
  //Dtype min;

  //std::cout << this->layer_param_.name() << std::endl;
  //std::cout << "top diff:";
  //max = top_diff[0];
  //min = top_diff[0];
  //sum = 0;
  //for(int i = 0; i < top[0]->count(); ++i) {
  //    if(max < top_diff[i]) max = top_diff[i];
  //    if(min > top_diff[i]) min = top_diff[i];
  //    sum += top_diff[i];
  //}
  //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/top[0]->count() << "]" << std::endl;
  //std::cout << std::endl;

  //std::cout << "bottom diff:";
  //max = bottom_diff[0];
  //min = bottom_diff[0];
  //sum = 0;
  //for(int i = 0; i < bottom[0]->count(); ++i) {
  //    if(max < bottom_diff[i]) max = bottom_diff[i];
  //    if(min > bottom_diff[i]) min = bottom_diff[i];
  //    sum += bottom_diff[i];
  //}
  //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/bottom[0]->count() << "]" << std::endl;
  //std::cout << std::endl;
  //}
  //print_count++;
  //added by ysm end

}

INSTANTIATE_CLASS(FlattenLayer);
REGISTER_LAYER_CLASS(Flatten);

}  // namespace caffe
