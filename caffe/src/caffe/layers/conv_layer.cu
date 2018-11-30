#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
		void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			//LOG(INFO)<<"convolution_layer:Forward_gpu";
			const Dtype* weight = this->blobs_[0]->gpu_data();
			for (int i = 0; i < bottom.size(); ++i) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* top_data = top[i]->mutable_gpu_data();
				for (int n = 0; n < this->num_; ++n) {
					this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
							top_data + n * this->top_dim_);
					if (this->bias_term_) {
						const Dtype* bias = this->blobs_[1]->gpu_data();
						this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
					}
				}
			}

		}

	template <typename Dtype>
		void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			const Dtype* weight = this->blobs_[0]->gpu_data();
			Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
			for (int i = 0; i < top.size(); ++i) {
				const Dtype* top_diff = top[i]->gpu_diff();
				// Bias gradient, if necessary.
				if (this->bias_term_ && this->param_propagate_down_[1]) {
					Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
					for (int n = 0; n < this->num_; ++n) {
						this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
					}
				}
				if (this->param_propagate_down_[0] || propagate_down[i]) {
					const Dtype* bottom_data = bottom[i]->gpu_data();
					Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
					for (int n = 0; n < this->num_; ++n) {
						// gradient w.r.t. weight. Note that we will accumulate diffs.
						if (this->param_propagate_down_[0]) {
							this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
									top_diff + n * this->top_dim_, weight_diff);
						}
						// gradient w.r.t. bottom data, if necessary.
						if (propagate_down[i]) {
							this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
									bottom_diff + n * this->bottom_dim_);
						}
					}
				}
			}
			//added by ysm
			//static int print_count = 0;
			//if(print_count%100 == 0) {
			//	if(this->layer_param_.name() == "weight_cccp5"||this->layer_param_.name() == "weight_cccp3"){
			//		std::cout << this->layer_param_.name() << std::endl;
			//		Dtype sum;
			//		Dtype max;
			//		Dtype min;
			//		int top_dim = top[0]->width()*top[0]->height();
			//		int bottom_dim = bottom[0]->width()*bottom[0]->height();
			//		int weight_dim = this->blobs_[0]->count()/this->blobs_[0]->num();
			//		
			//		std::cout << "topdata:";
			//		const Dtype* topdata = top[0]->cpu_data();
			//		sum = 0;
			//		max = topdata[0];
			//		min = topdata[0];
			//		for(int i = 0; i < top[0]->count(); ++i) {
			//			//std::cout << topdata[i] << " ";
			//			if(max < topdata[i]) max = topdata[i];
			//			if(min > topdata[i]) min = topdata[i];
			//			sum += topdata[i];
			//		}
			//		std::cout << "max:" << max << " min:" << min << " mean:" << sum/top[0]->count() << std::endl;
			//		std::cout << std::endl;
			//		
			//		std::cout << "bottomdata:";
			//		const Dtype* bottomdata = bottom[0]->cpu_data();
			//		sum = 0;
			//		max = bottomdata[0];
			//		min = bottomdata[0];
			//		for(int i = 0; i < bottom[0]->count(); ++i) {
			//			//std::cout << bottomdata[i] << " ";
			//			if(max < bottomdata[i]) max = bottomdata[i];
			//			if(min > bottomdata[i]) min = bottomdata[i];
			//			sum += bottomdata[i];
			//		}
			//		std::cout << "max:" << max << " min:" << min << " mean:" << sum/bottom[0]->count() << std::endl;
			//		std::cout << std::endl;
			//		
			//		std::cout << "topdiff:";
			//		const Dtype* topdiff = top[0]->cpu_diff();
			//		sum = 0;
			//		max = topdiff[0];
			//		min = topdiff[0];
			//		for(int i = 0; i < top[0]->count(); ++i) {
			//			//std::cout << topdiff[i] << " ";
			//			if(max < topdiff[i]) max = topdiff[i];
			//			if(min > topdiff[i]) min = topdiff[i];
			//			sum += topdiff[i];
			//		}
			//		std::cout << "max:" << max << " min:" << min << " mean:" << sum/top[0]->count() << std::endl;
			//		std::cout << std::endl;

			//		std::cout << "bottomdiff:";
			//		const Dtype* bottomdiff = bottom[0]->cpu_diff();
			//		sum = 0;
			//		max = bottomdiff[0];
			//		min = bottomdiff[0];
			//		for(int i = 0; i < bottom[0]->count(); ++i) {
			//			//std::cout << bottomdiff[i] << " ";
			//			if(max < bottomdiff[i]) max = bottomdiff[i];
			//			if(min > bottomdiff[i]) min = bottomdiff[i];
			//			sum += bottomdiff[i];
			//		}
			//		std::cout << "max:" << max << " min:" << min << " mean:" << sum/bottom[0]->count() << std::endl;
			//		std::cout << std::endl;

			//		std::cout << "weightdiff:";
			//		const Dtype* weightdiff = this->blobs_[0]->cpu_diff();
			//		sum = 0;
			//		max = weightdiff[0];
			//		min = weightdiff[0];
			//		for(int i = 0; i < this->blobs_[0]->count(); ++i) {
			//			//std::cout << weightdiff[i] << " ";
			//			if(max < weightdiff[i]) max = weightdiff[i];
			//			if(min > weightdiff[i]) min = weightdiff[i];
			//			sum += weightdiff[i];
			//		}
			//		std::cout << "max:" << max << " min:" << min << " mean:" << sum/this->blobs_[0]->count() << std::endl;
			//		std::cout << std::endl;
			//		
			//		std::cout << "weight:";
			//		const Dtype* weightdata = this->blobs_[0]->cpu_data();
			//		sum = 0;
			//		max = weightdata[0];
			//		min = weightdata[0];
			//		for(int i = 0; i < this->blobs_[0]->count(); ++i) {
			//			//std::cout << weightdiff[i] << " ";
			//			if(max < weightdata[i]) max = weightdata[i];
			//			if(min > weightdata[i]) min = weightdata[i];
			//			sum += weightdata[i];
			//		}
			//		std::cout << "max:" << max << " min:" << min << " mean:" << sum/this->blobs_[0]->count() << std::endl;
			//		std::cout << std::endl;
			//	}
			//}
			//if(this->layer_param_.name() == "weight_conv1")
			//	print_count++;
			//added by ysm end
		}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
