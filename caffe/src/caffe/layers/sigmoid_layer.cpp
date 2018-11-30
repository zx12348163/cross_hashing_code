#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
		inline Dtype sigmoid(Dtype x) {
			return 1. / (1. + exp(-x));
		}

	template <typename Dtype>
		void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = top[0]->mutable_cpu_data();
			const int count = bottom[0]->count();
			for (int i = 0; i < count; ++i) {
				top_data[i] = sigmoid(bottom_data[i]);
			}
			//added by ysm 
			//static int print_count = 0;
			//if(print_count%5 == 0) {
			//	int dim = bottom[0]->count()/bottom[0]->num();
			//	Dtype* bottom_temp = new Dtype[dim];
			//	Dtype* top_temp = new Dtype[dim];
			//	memset(bottom_temp, 0, sizeof(Dtype)*dim);
			//	for(int i = 0; i < 1; ++i) {
			//		for(int j = 0; j < dim; ++j) {
			//			bottom_temp[j] += bottom_data[i*dim+j];
			//			top_temp[j] += top_data[i*dim+j];
			//		}
			//	}
			//	std::cout << "sigmoidlayer bottom:";
			//	for(int i = 0; i < dim; ++i)	{
			//		//bottom_temp[i]/=bottom[0]->num();
			//		std::cout << bottom_temp[i] << " ";
			//	}
			//	std::cout << std::endl;
			//	std::cout << "sigmoidlayer top:";
			//	for(int i = 0; i < dim; ++i)	{
			//		//top_temp[i]/=top[0]->num();
			//		std::cout << top_temp[i] << " ";
			//	}
			//	std::cout << std::endl;
			//	delete []bottom_temp;
			//	delete []top_temp;
			//}
			//print_count++;
			//added by ysm end
		}

	template <typename Dtype>
		void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down,
				const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[0]) {
				const Dtype* top_data = top[0]->cpu_data();
				const Dtype* top_diff = top[0]->cpu_diff();
				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				const int count = bottom[0]->count();
				for (int i = 0; i < count; ++i) {
					const Dtype sigmoid_x = top_data[i];
					bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
				}
			}
		}

#ifdef CPU_ONLY
	STUB_GPU(SigmoidLayer);
#endif

	INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
