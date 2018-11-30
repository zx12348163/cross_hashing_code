#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void ChooselabelSoftmaxWithLossLayer<Dtype>::LayerSetUp(
							  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    LayerParameter  softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    softmax_bottom_vec_.clear();
    softmax_bottom_vec_.push_back(bottom[0]);
    softmax_top_vec_.clear();
    softmax_top_vec_.push_back(&prob_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
    if(top.size() >= 2){
      if(this->layer_param_.loss_weight_size() == 1) {
	this->layer_param_.add_loss_weight(Dtype(1.0));
      }
    }
  }
  

  template <typename Dtype>
  void ChooselabelSoftmaxWithLossLayer<Dtype>::Reshape(
						       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
    if (top.size() >= 2) {
      top[1]->ReshapeLike(*bottom[1]);
    }
  }

  template <typename Dtype>
  void ChooselabelSoftmaxWithLossLayer<Dtype>::Forward_cpu(
							   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the softmax prob values.
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* newLabel;
    if(top.size() >= 2){
      newLabel = top[1]->mutable_cpu_data(); 
      caffe_copy(top[1]->count(), label, newLabel);
    }
    int num = prob_.num();
    int dim = prob_.count() / num;
    int label_dim = bottom[1]->count() / bottom[1]->num();
    //DCHECK_EQ(dim, label_dim+1);
    Dtype loss = 0;
    for (int i = 0; i < num; ++i) {
      //if(label[i*label_dim] == -1)
      //	continue;
      //choose the best label
      int best_label = dim-1;
      Dtype best_prob = -1;
      Dtype sum_label = 0;
      for (int j = 0; j < label_dim; j++) {
	if(label[i*label_dim + j] <= 0)
	  continue;
	sum_label += label[i*label_dim + j];
	if(prob_data[i * dim + j] > best_prob){
	  best_label = j;
	  best_prob = prob_data[i*dim + j];
	}
      }
      if(top.size() == 2 && best_label < label_dim){
        newLabel[i*label_dim + best_label] = 0;
      }
      //if(top.size() ==2 && sum_label == 0)
      //newLabel[i*label_dim] = -1;
      // LOG(INFO) << "forward " << i;
      loss -= log(std::max(prob_data[i * dim + best_label],
			   Dtype(FLT_MIN))); 
    }
    top[0]->mutable_cpu_data()[0] = loss / num;
  }

  template <typename Dtype>
  void ChooselabelSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
							    const vector<bool>& propagate_down,
							    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      caffe_copy(prob_.count(), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->cpu_data();
      int num = prob_.num();
      int dim = prob_.count() / num;
      int label_dim = bottom[1]->count() / bottom[1]->num();
      for (int i = 0; i < num; ++i) {
	//LOG(INFO) << "back " << i;
	//choose the best label
	//if(label[i*label_dim] == -1){
	//  for(int c=0; c < dim; c++)
	//    bottom_diff[i*dim + c] = 0;
	//}else{
	int best_label = dim-1;
	Dtype best_prob = -1;
	for (int j = 0; j < label_dim; j++) {
	  if(label[i*label_dim + j] <= 0)
	    continue;
	  if(prob_data[i * dim + j] > best_prob){
	    best_label = j;
	    best_prob = prob_data[i*dim + j];
	  }
	}      
	bottom_diff[i * dim + best_label] -= 1;
	//}
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
    }
  }


#ifdef CPU_ONLY
  STUB_GPU(ChooselabelSoftmaxWithLossLayer);
#endif

  INSTANTIATE_CLASS(ChooselabelSoftmaxWithLossLayer);
  REGISTER_LAYER_CLASS(ChooselabelSoftmaxWithLoss);
}  // namespace caffe
