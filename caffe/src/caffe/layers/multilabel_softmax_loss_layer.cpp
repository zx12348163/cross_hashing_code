#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::LayerSetUp(
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
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int label_dim = bottom[1]->count() / bottom[1]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    Dtype sum_prob = 0.0;
    for (int j = 0; j < label_dim; j++) {
      sum_prob += label[i * label_dim + j];
    }
    for (int j = 0; j < label_dim; j++) {
      const Dtype label_value = (label[i * label_dim + j]) / sum_prob;
      loss -= label_value*log(std::max(prob_data[i * dim + j],
                           Dtype(FLT_MIN)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

  template <typename Dtype>
  void MultilabelSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
							 const vector<bool>& propagate_down,
							 const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      caffe_copy(prob_.count(), prob_data, bottom_diff);
      const Dtype* label = bottom[1]->cpu_data();
      int num = prob_.num();
      int dim = prob_.count() / num;
      int label_dim = bottom[1]->count() / bottom[1]->num();
      for (int i = 0; i < num; ++i) {
	Dtype sum_prob = 0.0;
	for (int j = 0; j < label_dim; j++) {
	  sum_prob += label[i * label_dim + j];
	}
	for (int j = 0; j < label_dim; ++j) {
	  const Dtype label_value = (label[i * label_dim + j]) / sum_prob;
	  bottom_diff[i * dim + j] -= label_value;
	}
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
    }
  }


#ifdef CPU_ONLY
STUB_GPU(MultilabelSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultilabelSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultilabelSoftmaxWithLoss);
}  // namespace caffe
