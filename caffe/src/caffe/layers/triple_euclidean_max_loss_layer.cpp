#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void TripleEuclideanMaxLossLayer<Dtype>::Reshape(
						 const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
    top[1]->Reshape(1, 1, 1, 1);
  }
  
  template <typename Dtype>
  void TripleEuclideanMaxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data_0 = bottom[0]->cpu_data();
    const Dtype* bottom_data_1 = bottom[1]->cpu_data();
    const Dtype* bottom_data_2 = bottom[2]->cpu_data();
    
    Dtype* diff_0 = bottom[0]->mutable_cpu_diff();
    memset(diff_0, Dtype(0), bottom[0]->count()*sizeof(Dtype));
    Dtype* diff_1 = bottom[1]->mutable_cpu_diff();
    memset(diff_1, Dtype(0), bottom[1]->count()*sizeof(Dtype));
    Dtype* diff_2 = bottom[2]->mutable_cpu_diff();
    memset(diff_2, Dtype(0), bottom[2]->count()*sizeof(Dtype));
    
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[1]->num(), bottom[2]->num());
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
    int num = bottom[0]->num();
    int dim = bottom[0]->channels();
    
    Dtype margin = this->layer_param_.contrastive_loss_param().margin();    
    Dtype loss = 0, acc = 0;
    for (int i = 0; i < num; ++i) {
        Dtype norm1 = 0, norm2 = 0;
      	for(int j = 0; j < dim; ++j){
	  norm1 += pow((bottom_data_0[i*dim + j] - bottom_data_1[i*dim + j]),2);
       	  norm2 += pow((bottom_data_0[i*dim + j] - bottom_data_2[i*dim + j]),2);
	}
	if(margin + norm1 - norm2 > 0){
	  loss += (margin + norm1 - norm2);
	  for(int j = 0; j < dim; ++j){
	    diff_0[i*dim +j] += 2*(bottom_data_2[i*dim + j] - bottom_data_1[i*dim + j]);
	    diff_1[i*dim +j] += 2*(bottom_data_1[i*dim + j] - bottom_data_0[i*dim + j]);
	    diff_2[i*dim +j] += 2*(bottom_data_0[i*dim + j] - bottom_data_2[i*dim + j]);
	  }
	}
	else ++ acc;
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    top[0]->mutable_cpu_data()[0] = loss / num;
    if (phase_ == "thief") {
      caffe_scal(bottom[0]->count(), loss_weight / num / margin, diff_0);
      caffe_scal(bottom[1]->count(), loss_weight / num / margin, diff_1);
      caffe_scal(bottom[2]->count(), loss_weight / num / margin, diff_2);
    }
    else {
      caffe_scal(bottom[0]->count(), Dtype(0), diff_0);
      caffe_scal(bottom[1]->count(), Dtype(0), diff_1);
      caffe_scal(bottom[2]->count(), Dtype(0), diff_2);
    }
    top[1]->mutable_cpu_data()[0] = acc / num;
  }

  template <typename Dtype>
  void TripleEuclideanMaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  }

#ifdef CPU_ONLY
  STUB_GPU(TripleEuclideanMaxLossLayer);
#endif

  INSTANTIATE_CLASS(TripleEuclideanMaxLossLayer);
  REGISTER_LAYER_CLASS(TripleEuclideanMaxLoss);

}  // namespace caffe
