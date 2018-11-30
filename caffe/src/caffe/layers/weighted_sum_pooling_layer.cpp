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
		void WeightedSumPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			CHECK_EQ(4, bottom[0]->num_axes());
			//CHECK_EQ(2, bottom[1]->num_axes());
			channels_ = bottom[0]->channels();
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			weight_num_ = bottom[1]->count(1);
			CHECK_EQ(weight_num_, height_ * width_);
		}

	template <typename Dtype>
		void WeightedSumPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			CHECK_EQ(4, bottom[0]->num_axes());
			//CHECK_EQ(2, bottom[1]->num_axes());
			CHECK_EQ(channels_, bottom[0]->channels());
			CHECK_EQ(height_, bottom[0]->height());
			CHECK_EQ(width_, bottom[0]->width());
			CHECK_EQ(weight_num_, bottom[1]->count(1));

			vector<int> top_shape;
			top_shape.push_back(bottom[0]->num());
			top_shape.push_back(channels_);
			top[0]->Reshape(top_shape);


		}

	template <typename Dtype>
		void WeightedSumPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* weights_data = bottom[1]->cpu_data();
			Dtype* top_data = top[0]->mutable_cpu_data();
			for(int n = 0; n < bottom[0]->num(); ++n) {
				for(int c = 0; c < channels_; ++c) {
					Dtype sum = 0;
					for(int i = 0; i < weight_num_; ++i) {
						sum += weights_data[i] * bottom_data[i];
					}
					top_data[c] = sum;
					bottom_data += bottom[0]->offset(0, 1);
				}
				weights_data += bottom[1]->offset(1);
				top_data += top[0]->offset(1);
			}

			//added by ysm 
			//static int print_count = 0;
			//if(print_count%100 == 0) {
			//	top_data = top[0]->mutable_cpu_data();
			//	bottom_data = bottom[0]->cpu_data();
			//	weights_data = bottom[1]->cpu_data();
			//	int top_dim = top[0]->count()/top[0]->num();
			//	std::cout << "top:";
			//	for(int i = 0; i < top_dim; ++i) {
			//		std::cout << top_data[i] << " ";
			//	}
			//	std::cout << std::endl;


			//	std::cout << "map:";
			//	for(int i = 0; i < weight_num_; ++i) {
			//		std::cout << bottom_data[i] << " ";
			//	}
			//	std::cout << std::endl;

			//	Dtype sum = 0;
			//	Dtype top_sum = 0;
			//	std::cout << "weight:";
			//	for(int i = 0; i < weight_num_; ++i) {
			//		std::cout << weights_data[i] << " ";
			//		sum += weights_data[i];
			//		top_sum += weights_data[i]*bottom_data[i];
			//	}
			//	std::cout << std::endl;
			//	std::cout << "weight sum:" << sum << std::endl;
			//	std::cout << "top sum:" << top_sum << std::endl;
			//}
			//print_count++;
			//added by ysm end 
		}


  template <typename Dtype>
  void WeightedSumPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
      for(int wn = 0; wn < weight_num_; ++wn) {
	Dtype sum = 0;
	for(int c = 0; c < channels_; ++c) {
	  int index = wn + c * weight_num_;
	  sum += top_diff[c] * bottom_data[index];
	}
	weights_diff[wn] += sum;
	//weights_diff[wn] = 0.00001;
      }

      for(int c = 0; c < channels_; ++c) {
	for(int i = 0; i < weight_num_; ++i) {
	  bottom_diff[i] += top_diff[c] * weights_data[i];
	  //bottom_diff[i] = 0.00001;
	}
	bottom_diff += bottom[0]->offset(0,1);
      }

      top_diff += top[0]->offset(1);
      bottom_data += bottom[0]->offset(1);
      weights_diff += bottom[1]->offset(1);
      weights_data += bottom[1]->offset(1);
    } 
    //added by ysm
    if(print_count%500 == 0) {
      //std::cout << "top offset(1):" << top[0]->offset(1) << std::endl;
      //std::cout << "bottom[0] offset(1):" << bottom[0]->offset(1) << std::endl;
      //std::cout << "bottom[1] offset(1):" << bottom[1]->offset(1) << std::endl;
      //std::cout << "top offset(0,1):" << top[0]->offset(0,1) << std::endl;
      //std::cout << "bottom[0] offset(0,1):" << bottom[0]->offset(0,1) << std::endl;
      //std::cout << "bottom[1] offset(0,1):" << bottom[1]->offset(0,1) << std::endl;
      std::cout << this->layer_param_.name() << std::endl;
      top_diff = top[0]->cpu_diff();
      bottom_diff = bottom[0]->mutable_cpu_diff();
      weights_diff = bottom[1]->mutable_cpu_diff();
      bottom_data = bottom[0]->cpu_data();
      weights_data = bottom[1]->cpu_data();
      const Dtype* top_data = top[0]->cpu_data();
      Dtype sum;
      Dtype max;
      Dtype min;
				
      //int top_dim = top[0]->count()/top[0]->num();
      //std::cout << "top:";
      //sum = 0;
      //max = top_data[0];
      //min = top_data[0];
      //for(int i = 0; i < top[0]->count(); ++i) {
      //	//std::cout << top_data[i] << " ";
      //	if(max < top_data[i]) max = top_data[i];
      //	if(min > top_data[i]) min = top_data[i];
      //	sum += top_data[i];
      //}
      //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/top[0]->count() << "]" << std::endl;
      //std::cout << std::endl;

      //std::cout << "map:";
      //sum = 0;
      //max = bottom_data[0];
      //min = bottom_data[0];
      //for(int i = 0; i < bottom[0]->count(); ++i) {
      //	//std::cout << bottom_data[i] << " ";
      //	if(max < bottom_data[i]) max = bottom_data[i];
      //	if(min > bottom_data[i]) min = bottom_data[i];
      //	sum += bottom_data[i];
      //}
      //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/bottom[0]->count() << "]" << std::endl;
      //std::cout << std::endl;

      std::cout << "weight:";
      sum = 0;
      max = weights_data[0];
      min = weights_data[0];
      for(int i = 0; i < bottom[1]->count(); ++i) {
	//std::cout << weights_data[i] << " ";
	//sum += weights_data[i];
	//top_sum += weights_data[i]*bottom_data[i];
	if(max < weights_data[i]) max = weights_data[i];
	if(min > weights_data[i]) min = weights_data[i];
	sum += weights_data[i];
      }
      std::cout << "[max:" << max << " min:" << min << " mean:" << sum/bottom[1]->count() << "]" << std::endl;
      std::cout << std::endl;
				
      //std::cout << "topdiff:";
      //sum = 0;
      //max = top_diff[0];
      //min = top_diff[0];
      //for(int i = 0; i < top[0]->count(); ++i) {
      //	//std::cout << top_diff[i] << " ";
      //	if(max < top_diff[i]) max = top_diff[i];
      //	if(min > top_diff[i]) min = top_diff[i];
      //	sum += top_diff[i];
      //}
      //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/top[0]->count() << "]" << std::endl;
      //std::cout << std::endl;

      //std::cout << "mapdiff:";
      //sum = 0;
      //max = bottom_diff[0];
      //min = bottom_diff[0];
      //int data_dim = bottom[0]->count()/bottom[0]->num();
      //for(int i = 0; i < bottom[0]->count(); ++i) {
      //	//std::cout << bottom_diff[i] << " ";
      //	if(max < bottom_diff[i]) max = bottom_diff[i];
      //	if(min > bottom_diff[i]) min = bottom_diff[i];
      //	sum += bottom_diff[i];
      //}
      //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/bottom[0]->count() << "]" << std::endl;
      //std::cout << std::endl;

      //std::cout << "weightdiff:";
      //sum = 0;
      //max = weights_diff[0];
      //min = weights_diff[0];
      //int weight_dim = bottom[1]->count()/bottom[1]->num();
      //for(int i = 0; i < bottom[1]->count(); ++i) {
      //	//std::cout << weights_diff[i] << " ";
      //	if(max < weights_diff[i]) max = weights_diff[i];
      //	if(min > weights_diff[i]) min = weights_diff[i];
      //	sum += weights_diff[i];
      //}
      //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/bottom[1]->count() << "]" << std::endl;
      //std::cout << std::endl;

    }
    print_count++;
    //added by ysm end

  }

#ifdef CPU_ONLY
	STUB_GPU(PoolingLayer);
#endif

	INSTANTIATE_CLASS(WeightedSumPoolingLayer);
	REGISTER_LAYER_CLASS(WeightedSumPooling);

}  // namespace caffe
