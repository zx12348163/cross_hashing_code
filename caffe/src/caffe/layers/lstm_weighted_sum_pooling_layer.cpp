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
		void LstmWeightedSumPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			CHECK_EQ(4, bottom[0]->num_axes());
			//CHECK_EQ(2, bottom[1]->num_axes());
			channels_ = bottom[0]->channels();
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			weight_num_ = bottom[1]->count(2);
			CHECK_EQ(weight_num_, height_ * width_);
		}

	template <typename Dtype>
		void LstmWeightedSumPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			CHECK_EQ(4, bottom[0]->num_axes());
			//CHECK_EQ(2, bottom[1]->num_axes());
			CHECK_EQ(channels_, bottom[0]->channels());
			CHECK_EQ(height_, bottom[0]->height());
			CHECK_EQ(width_, bottom[0]->width());
			CHECK_EQ(weight_num_, bottom[1]->count(2));

			vector<int> top_shape;
			top_shape.push_back(bottom[1]->shape(0));
			top_shape.push_back(bottom[1]->shape(1));
			top_shape.push_back(channels_);
			top[0]->Reshape(top_shape);


		}

	template <typename Dtype>
		void LstmWeightedSumPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const int T = bottom[1]->shape(0);
			const int N = bottom[1]->shape(1);
			const int img_dim = bottom[0]->count(1);
			const int weight_dim = bottom[1]->count(1);
			const int top_dim = top[0]->count(1);
			for(int t = 0; t < T; ++t) {
				for(int n = 0; n < N; ++n) {
					const Dtype* img_data = bottom[0]->cpu_data() + n * img_dim;
					const Dtype* weight_data = bottom[1]->cpu_data() + t * weight_dim + n * weight_num_;
					Dtype* top_data = top[0]->mutable_cpu_data() + t * top_dim + n * channels_;
					caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_, weight_num_, 
							(Dtype)1., img_data, weight_data, (Dtype)0., top_data);
				}
			}
		}


	template <typename Dtype>
		void LstmWeightedSumPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			const int T = bottom[1]->shape(0);
			const int N = bottom[1]->shape(1);
			const int img_dim = bottom[0]->count(1);
			const int weight_dim = bottom[1]->count(1);
			const int top_dim = top[0]->count(1);
			caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
			for(int t = 0; t < T; ++t) {
				for(int n = 0; n < N; ++n) {
					const Dtype* img_data = bottom[0]->cpu_data() + n * img_dim;
					const Dtype* weight_data = bottom[1]->cpu_data() + t * weight_dim + n * weight_num_;
					Dtype* img_diff = bottom[0]->mutable_cpu_diff() + n *img_dim;
					Dtype* weight_diff = bottom[1]->mutable_cpu_diff() + t * weight_dim + n * weight_num_;
					const Dtype* top_diff = top[0]->cpu_diff() + t * top_dim + n * channels_;
					caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, weight_num_, 1, 
							channels_, Dtype(1), img_data, top_diff, Dtype(0), weight_diff);
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, weight_num_,
							1, Dtype(1), top_diff, weight_data, Dtype(1), img_diff);
				}
			}
		}

#ifdef CPU_ONLY
	STUB_GPU(LstmWeightedSumPoolingLayer);
#endif

	INSTANTIATE_CLASS(LstmWeightedSumPoolingLayer);
	REGISTER_LAYER_CLASS(LstmWeightedSumPooling);

}  // namespace caffe
