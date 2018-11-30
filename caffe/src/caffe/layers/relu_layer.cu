#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
		__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
				Dtype negative_slope) {
			CUDA_KERNEL_LOOP(index, n) {
				out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
			}
		}

	template <typename Dtype>
		void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = top[0]->mutable_gpu_data();
			const int count = bottom[0]->count();
			Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
			// NOLINT_NEXT_LINE(whitespace/operators)
			ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, top_data, negative_slope);
			CUDA_POST_KERNEL_CHECK;
			// << " count: " << count << " bottom_data: "
			//     << (unsigned long)bottom_data
			//     << " top_data: " << (unsigned long)top_data
			//     << " blocks: " << CAFFE_GET_BLOCKS(count)
			//     << " threads: " << CAFFE_CUDA_NUM_THREADS;
		}

	template <typename Dtype>
		__global__ void ReLUBackward(const int n, const Dtype* in_diff,
				const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
			CUDA_KERNEL_LOOP(index, n) {
				out_diff[index] = in_diff[index] * ((in_data[index] > 0)
						+ (in_data[index] <= 0) * negative_slope);
			}
		}

	template <typename Dtype>
		void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down,
				const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[0]) {
				const Dtype* bottom_data = bottom[0]->gpu_data();
				const Dtype* top_diff = top[0]->gpu_diff();
				Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
				const int count = bottom[0]->count();
				Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
				// NOLINT_NEXT_LINE(whitespace/operators)
				ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
						count, top_diff, bottom_data, bottom_diff, negative_slope);
				CUDA_POST_KERNEL_CHECK;
			}
			//added by ysm 
			//static int print_count = 0;
			//if(print_count%100 == 0) {
			//	if(this->layer_param_.name() == "relu3") {
			//		std::cout << this->layer_param_.name() << std::endl;
			//		const Dtype* topdiff = top[0]->cpu_diff();
			//		const Dtype* bottomdiff = bottom[0]->cpu_diff();
			//		std::cout << "top diff:";
			//		int dim = top[0]->width()*top[0]->height();
			//		for(int i = 0; i < dim; ++i) {
			//			std::cout << topdiff[i] << " ";
			//		}
			//		std::cout << std::endl;
			//		std::cout << "bottom diff:";
			//		for(int i = 0; i < dim; ++i) {
			//			std::cout << bottomdiff[i] << " ";
			//		}
			//		std::cout << std::endl;
			//	}
			//}
			//if(this->layer_param_.name() == "relu3")
			//	print_count++;
			//added by ysm end
		}


	INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
