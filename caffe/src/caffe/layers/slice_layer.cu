#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Backward_cpu(top, propagate_down, bottom);
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff);
    offset_slice_axis += top_slice_axis;
  }
  //added by ysm 
  //static int print_count = 0;
  //if(print_count%100 == 0) {
  //const Dtype* bottom_diff = bottom[0]->cpu_diff();
  //const Dtype* weight_diff = top[0]->cpu_diff();
  //const Dtype* map_diff = top[1]->cpu_diff();
  //Dtype sum;
  //Dtype max;
  //Dtype min;

  //std::cout << this->layer_param_.name() << std::endl;
  //std::cout << "bottom diff:";
  //max = bottom_diff[0];
  //min = bottom_diff[0];
  //sum = 0;
  //for(int i = 0; i < bottom[0]->count(); ++i) {
  //    if(max < bottom_diff[i]) max = bottom_diff[i];
  //    if(min > bottom_diff[i]) min = bottom_diff[i];
  //    sum += bottom_diff[i];
  //}
  //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/bottom[0]->count() << "]" << std::endl;
  //std::cout << std::endl;

  //std::cout << "top[0] diff:";
  //max = weight_diff[0];
  //min = weight_diff[0];
  //sum = 0;
  //for(int i = 0; i < top[0]->count(); ++i) {
  //    if(max < weight_diff[i]) max = weight_diff[i];
  //    if(min > weight_diff[i]) min = weight_diff[i];
  //    sum += weight_diff[i];
  //}
  //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/top[0]->count() << "]" << std::endl;
  //std::cout << std::endl;
  //
  //std::cout << "top[1] diff:";
  //max = map_diff[0];
  //min = map_diff[0];
  //sum = 0;
  //for(int i = 0; i < top[1]->count(); ++i) {
  //    if(max < map_diff[i]) max = map_diff[i];
  //    if(min > map_diff[i]) min = map_diff[i];
  //    sum += map_diff[i];
  //}
  //std::cout << "[max:" << max << " min:" << min << " mean:" << sum/top[1]->count() << "]" << std::endl;
  //std::cout << std::endl;
  //}
  //print_count++;
  //added by ysm end
}

INSTANTIATE_LAYER_GPU_FUNCS(SliceLayer);

}  // namespace caffe
