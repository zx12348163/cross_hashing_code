#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime> 
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
int PairContrastiveLossLayer<Dtype>::similarity(const Dtype* label,int i, int j, int label_dim) {
	for(int k = 0; k < label_dim; ++k) {
		if(1 == label[i*label_dim + k] && label[i*label_dim + k] == label[j*label_dim + k]) {
			return 1;
		}
	}
	return 0;
}

template <typename Dtype>
void PairContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void PairContrastiveLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num() * 2);
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  top[1]->Reshape(loss_shape);
}

template <typename Dtype>
void PairContrastiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label_data = bottom[1]->cpu_data();
	const int batch_size = bottom[0]->num() / 2;
	CHECK(bottom[0]->num() % 2 == 0);
	const int data_dim = bottom[0]->count() / bottom[0]->num();
	const Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	const Dtype sim_margin = this->layer_param_.contrastive_loss_param().sim_margin();
	int similar_count = 0;
	int dissimilar_count = 0;
	int similar_acc = 0;
	int dissimilar_acc = 0;
	Dtype loss = 0;
	for(int i = 0; i < batch_size; ++i) {
		Dtype dist = 0;
		for(int k = 0; k < data_dim; ++k) {
			//dist += (bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k])*(bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);
			dist += fabs(bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);
		}				
		//dist = sqrt(dist);
		//construct data pair
		if(label_data[i]) { //similar pair
			++similar_count;
			//loss += std::max(dist - sim_margin, Dtype(0)) * std::max(dist - sim_margin, Dtype(0));
			loss += std::max(dist - sim_margin, Dtype(0.0));
			if(dist < sim_margin)
				similar_acc++;
				//similar_dist += dist;
		} else { // dismilar pair
			++dissimilar_count;
			//loss += std::max(margin - dist, Dtype(0)) * std::max(margin - dist, Dtype(0));
			loss += std::max(margin - dist, Dtype(0.0));
			if(dist > margin)
				dissimilar_acc++;
				//dissimilar_dist += dist;
		}
	}
	loss = loss / static_cast<Dtype>(batch_size);
	top[0]->mutable_cpu_data()[0] = loss;
	top[1]->mutable_cpu_data()[0] = (similar_acc + dissimilar_acc) / static_cast<Dtype>(batch_size);
	//std::cout << "similar: " << similar_count <<  std::endl 
	//	<< "dissimilar: " << dissimilar_count << std::endl;
	//std::cout << "similar dist:" << similar_dist / similar_pairs_.size() << std::endl
	//	<< "dissimilar dist:" << dissimilar_dist / dissimilar_pairs_.size() << std::endl;
}
template <typename Dtype>
void PairContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label_data = bottom[1]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int batch_size = bottom[0]->num() / 2;
	CHECK(bottom[0]->num() % 2 == 0);
	const int data_dim = bottom[0]->count() / bottom[0]->num();
  	const Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  	const Dtype sim_margin = this->layer_param_.contrastive_loss_param().sim_margin();
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	for(int i = 0; i < batch_size; ++i) {
		Dtype dist = 0;
		for(int k = 0; k < data_dim; ++k) {
			dist += fabs(bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);
			//dist += (bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k])*(bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);			
		}
		//dist = sqrt(dist);
		if(label_data[i]) { // similar pair
			if(dist > sim_margin) {
				//Dtype beta = (dist - sim_margin) / (dist + Dtype(1e-4));
				for(int k = 0; k < data_dim; ++k) {
					//bottom_diff[(2*i)*data_dim + k] = (bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);
					//bottom_diff[(2*i+1)*data_dim + k] = -(bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);
					if(bottom_data[(2*i)*data_dim + k] > bottom_data[(2*i+1)*data_dim + k]) {
						bottom_diff[(2*i)*data_dim + k] += 1;
						bottom_diff[(2*i+1)*data_dim + k] += -1;
					} else if(bottom_data[(2*i)*data_dim + k] < bottom_data[(2*i+1)*data_dim + k]){
						bottom_diff[(2*i)*data_dim + k] += -1;
						bottom_diff[(2*i+1)*data_dim + k] += 1;
					}

					
				}
			}
		} else { // dissimilar
			if(dist < margin) {
				//Dtype beta = -(margin - dist) / (dist + Dtype(1e-4));
				for(int k = 0; k < data_dim; ++k) {
					//bottom_diff[(2*i)*data_dim + k] = -(bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]);
					//bottom_diff[(2*i+1)*data_dim + k] = (bottom_data[(2*i)*data_dim + k] - bottom_data[(2*i+1)*data_dim + k]); 
					if(bottom_data[(2*i)*data_dim + k] > bottom_data[(2*i+1)*data_dim + k]) {
						bottom_diff[(2*i)*data_dim + k] += -1;
						bottom_diff[(2*i+1)*data_dim + k] += 1;
					} else if(bottom_data[(2*i)+data_dim + k] < bottom_data[(2*i+1)*data_dim + k]){
						bottom_diff[(2*i)*data_dim + k] += 1;
						bottom_diff[(2*i+1)*data_dim + k] += -1;
					}
				}
			}
		}
	}
	caffe_scal(bottom[0]->count(), static_cast<Dtype>(top[0]->cpu_diff()[0]) / static_cast<Dtype>(batch_size), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(PairContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(PairContrastiveLossLayer);
REGISTER_LAYER_CLASS(PairContrastiveLoss);

}  // namespace caffe


