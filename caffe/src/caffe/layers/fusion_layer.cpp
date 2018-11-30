// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {


  template <typename Dtype>
  void FusionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					  const vector<Blob<Dtype>*>& top) {
    patch_num_ = bottom[0]->num();
    patch_dim_ = bottom[0]->count() / bottom[0]->num();
    label_num_ = bottom[1]->count() / bottom[1]->num();
    out_dim_each_img_ = patch_dim_;
    (top)[0]->Reshape(patch_num_, out_dim_each_img_, 1, 1);
  }

  // Forward_cpu for FusionLayer
  template <typename Dtype>
  void FusionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
					   const vector<Blob<Dtype>*>& top){
   
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = (top)[0]->mutable_cpu_data();
    //Different patches are fusing into one image via max_pooling
    const Dtype* score_mat = bottom[1]->cpu_data();
  
    // Initialize
    memset(top_data, 0, top[0]->count()*sizeof(Dtype));
   
    // The main loop
    for (int n = 0; n < patch_num_; ++n){
      int max_label = 0;
      Dtype max_prob = -1;
      for(int la=0; la < label_num_; la++){
	if(score_mat[n*label_num_ + la] > max_prob){
	  max_prob = score_mat[n*label_num_ + la];
          max_label = la;
	}
      }
      if(max_label == label_num_ - 1)
	continue;
      for (int d = 0; d < patch_dim_; ++d ){
	top_data[n*patch_dim_ + d] += bottom_data[n*patch_dim_+d];	 	      	  
      }	
    }

  }


  //Backward_cpu for FusionLayer
  template <typename Dtype>
  void FusionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
					    const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
 
    if (!propagate_down[0]){
      //return Dtype(0.);
    }
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* bottom_data = (bottom)[0]->cpu_data();
    Dtype* bottom_diff = (bottom)[0]->mutable_cpu_diff();
    const Dtype* score_mat = bottom[1]->cpu_data();
    memset(bottom_diff, 0, (bottom)[0]->count() * sizeof(Dtype));
    Dtype* score_diff = (bottom)[1]->mutable_cpu_diff();
    memset(score_diff, 0, (bottom)[1]->count() * sizeof(Dtype));


    for (int n = 0; n < patch_num_; ++n){
      int max_label = 0;
      Dtype max_prob = -1;
      for(int la=0; la < label_num_; la++){
	if(score_mat[n*label_num_ + la] > max_prob){
	  max_prob = score_mat[n*label_num_ + la];
          max_label = la;
	}
      }
      if(max_label == label_num_ - 1)
	continue;    
      for(int d = 0; d < patch_dim_; ++d){
	bottom_diff[n*patch_dim_ + d] += top_diff[n*patch_dim_ + d];	 
      }
    }
  }				
  
				

    
#ifdef CPU_ONLY
  STUB_GPU(FusionLayer);
#endif

  INSTANTIATE_CLASS(FusionLayer);
  REGISTER_LAYER_CLASS(Fusion);
  
  //  INSTANTIATE_CLASS(HashFusionLayer);


}  // namespace caffe
