// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

//#include "caffe/common.hpp"
#include "caffe/layer.hpp"
//#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
//#include "caffe/util/io.hpp"

namespace caffe {

  template <typename Dtype>
  void DropWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
    threshold_ = this->layer_param_.dropout_param().dropout_ratio();
    DCHECK(threshold_ > 0.);
    DCHECK(threshold_ < 1.);
  }

  template <typename Dtype>
  void DropWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
  }

  template <typename Dtype>
  void DropWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					   const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    memset(top_data, Dtype(0.0), sizeof(Dtype)*top[0]->count());
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const int ratio = int(dim * threshold_);
    for (int i = 0; i < num; ++i) {
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < dim; ++k) {
        bottom_data_vector.push_back(std::make_pair(
						    bottom_data[i * dim + k], k));
      }
      std::partial_sort(bottom_data_vector.begin(), bottom_data_vector.begin() + ratio, bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      for(int k=0; k < ratio; k++)
	top_data[i*dim + bottom_data_vector[k].second] = bottom_data[i*dim + bottom_data_vector[k].second];
    }
  }

  template <typename Dtype>
  void DropWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					    const vector<bool>& propagate_down,
					    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      memset(bottom_diff, Dtype(0.0), sizeof(Dtype)*bottom[0]->count());
      const int num = bottom[0]->num();
      const int dim = bottom[0]->count() / num;
      const int ratio = int(dim * threshold_);
      for (int i = 0; i < num; ++i) {
	std::vector<std::pair<Dtype, int> > bottom_data_vector;
	for (int k = 0; k < dim; ++k) {
	  bottom_data_vector.push_back(std::make_pair(
						      bottom_data[i * dim + k], k));
	}
        std::partial_sort(bottom_data_vector.begin(), bottom_data_vector.begin() + ratio,
			  bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
        for(int k=0; k < ratio; k++)
          bottom_diff[i*dim + bottom_data_vector[k].second] = top_diff[i*dim + bottom_data_vector[k].second];
      }
    }
  }


#ifdef CPU_ONLY
  STUB_GPU(DropWeightLayer);
#endif

  INSTANTIATE_CLASS(DropWeightLayer);
  REGISTER_LAYER_CLASS(DropWeight);

}  // namespace caffe
