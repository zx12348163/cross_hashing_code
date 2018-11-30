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
  __global__ void AveForwardLayer(const int nthread, const int n_proposal, const int label_num, const int patch_dim, const Dtype* bottom_data, const Dtype* score_mat, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthread) {
      int d = index % patch_dim;
      int la = (index / patch_dim) % label_num;
      int p = (index / patch_dim / label_num) % n_proposal;
      top_data[la*patch_dim + d] += score_mat[p*label_num + la] * bottom_data[p*patch_dim + d]/n_proposal;
    }
  }


  template <typename Dtype>
  __global__ void AveBackwardLayer(const int nthread, const int n_proposal, const int label_num, const int patch_dim, const Dtype* top_diff, const Dtype* score_mat, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthread) {
      int p = index % n_proposal;
      int la = (index / n_proposal) % label_num;
      int d = (index / n_proposal / label_num) % patch_dim;
      bottom_diff[p*patch_dim + d] += score_mat[p*label_num + la] * top_diff[la*patch_dim + d]/n_proposal;
    }
  }



  // Forward_cpu for FusionLayer
  template <typename Dtype>
  void FusionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
					   const vector<Blob<Dtype>*>& top){
    // const Dtype* bottom_data = bottom[0]->gpu_data();
    // Dtype* top_data = (top)[0]->mutable_cpu_data();
    // //Different patches are fusing into one image via max_pooling
    // const Dtype* score_mat = bottom[1]->gpu_data();
    // const Dtype* conv5_scales = bottom[2]->cpu_data();
    // const int n_scales = bottom[2]->channels();
    // caffe_set(top[0]->count(), Dtype(0), top_data);
    // top_data = (top)[0]->mutable_gpu_data();
   
    // for (int n = 0; n < img_num_; ++n){
    //   int n_proposal = conv5_scales[n*n_scales];
    //   int nthread = n_proposal * label_num_ * patch_dim_;
    //   AveForwardLayer<<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
    // 									     nthread, n_proposal, label_num_, patch_dim_, bottom_data,score_mat, top_data);

    //   bottom_data += bottom[0]->offset(patch_num_each_img_);
    //   score_mat +=  bottom[1]->offset(patch_num_each_img_);
    //   top_data += (top)[0]->offset(1);
    // }
     Forward_cpu(bottom, top);
  }
  // Backward_cpu for FusionLayer
  template <typename Dtype>
  void FusionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
					    const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
    Backward_cpu(top, propagate_down, bottom);
    // if (!propagate_down[0]){
    //   //return Dtype(0.);
    // }
    // const Dtype* top_diff = top[0]->gpu_diff();
    // Dtype* bottom_diff = (bottom)[0]->mutable_cpu_diff();
    // const Dtype* score_mat = bottom[1]->gpu_data();
    // memset(bottom_diff, 0, (bottom)[0]->count() * sizeof(Dtype));
    // bottom_diff = (bottom)[0]->mutable_gpu_diff();
    // const Dtype* conv5_scales = bottom[2]->cpu_data();
    // const int n_scales = bottom[2]->channels();
    // for (int n = 0; n < img_num_; ++n){
    //   int n_proposal = conv5_scales[n*n_scales];
    //   int nthread = n_proposal * label_num_ * patch_dim_;
    //   AveBackwardLayer<<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
    // 									      nthread, n_proposal, label_num_, patch_dim_,top_diff, score_mat, bottom_diff);
    //   score_mat += (bottom)[1]->offset(patch_num_each_img_);
    //   bottom_diff += (bottom)[0]->offset(patch_num_each_img_);
    //   top_diff += top[0]->offset(1);
    // }
  }
  
  INSTANTIATE_LAYER_GPU_FUNCS(FusionLayer);
}  // namespace caffe
