#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
		void FCWeightedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			CHECK_EQ(2, bottom[0]->num_axes());
			//CHECK_EQ(2, bottom[1]->num_axes());
			channels_ = bottom[0]->channels();
			weight_num_ = bottom[1]->count(1);
			CHECK_EQ(weight_num_, channels_);
		}

	template <typename Dtype>
		void FCWeightedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			top[0]->ReshapeLike(*bottom[0]);
		}

	template <typename Dtype>
		void FCWeightedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* weights_data = bottom[1]->cpu_data();
			Dtype* top_data = top[0]->mutable_cpu_data();
			for(int n = 0; n < bottom[0]->num(); ++n) {
				for(int c = 0; c < channels_; ++c) {
					top_data[c] = weights_data[c] * bottom_data[c];
				}
				weights_data += bottom[1]->offset(1);
				bottom_data += bottom[0]->offset(1);
				top_data += top[0]->offset(1);
			}
		}


	template <typename Dtype>
		void FCWeightedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			static int print_count = 0;
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* weights_data = bottom[1]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			Dtype* weights_diff = bottom[1]->mutable_cpu_diff();
			caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
			caffe_set(bottom[1]->count(), Dtype(0), weights_diff);
			//memset(bottom_diff, 0, sizeof(Dtype)*bottom[0]->count());
			//memset(weights_diff, 0, sizeof(Dtype)*bottom[1]->count());
			for(int n = 0; n < top[0]->num(); ++n) {
				for(int c = 0; c < channels_; ++c) {
					bottom_diff[c] = top_diff[c] * weights_data[c];
					weights_diff[c] = top_diff[c] * bottom_data[c];
				}
				bottom_diff += bottom[0]->offset(1);
				bottom_data += bottom[0]->offset(1);
				weights_diff += bottom[1]->offset(1);
				weights_data += bottom[1]->offset(1);
				top_diff += top[0]->offset(1);

			} 
		}

#ifdef CPU_ONLY
	STUB_GPU(FCWeightedLayer);
#endif

	INSTANTIATE_CLASS(FCWeightedLayer);
	REGISTER_LAYER_CLASS(FCWeighted);

}  // namespace caffe
