#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void CrossModalLossLayer<Dtype>::Reshape(
						 const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
    top[1]->Reshape(1, 1, 1, 1);
  }


  template <typename Dtype>
  Dtype CrossModalLossLayer<Dtype>::Similarity(
						     const Dtype* labels, int i, int j, const int dim, const int batch_size) {
    Dtype s_sim = 0;
    i %= batch_size;
    j %= batch_size;
    if (i == j) {
        return dim;
    }
    //return 0; //// for unsupervised
    for(int k = 0; k < dim; ++ k){
	    s_sim += labels[i*dim+k] * labels[j*dim+k];
    }
    return s_sim;
  }

  template <typename Dtype>
  void CrossModalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data_0 = bottom[0]->cpu_data();
    const Dtype* bottom_data_1 = bottom[1]->cpu_data();
    const Dtype* label = bottom[2]->cpu_data();
    
    Dtype* diff_0 = bottom[0]->mutable_cpu_diff();
    memset(diff_0, Dtype(0), bottom[0]->count()*sizeof(Dtype));
    Dtype* diff_1 = bottom[1]->mutable_cpu_diff();
    memset(diff_1, Dtype(0), bottom[1]->count()*sizeof(Dtype));
    
    int num_0 = bottom[0]->num();
    int num_1 = bottom[1]->num();
    CHECK_EQ(bottom[0]->count() / num_0, bottom[1]->count() / num_1);
    int dim = bottom[0]->count() / num_0;
    int label_dim = bottom[2]->count() / bottom[2]->num();
    
    Dtype margin = this->layer_param_.contrastive_loss_param().margin();    
    Dtype loss = 0, n_tri = 0, acc = 0;
    for (int i = 0; i < num_0; ++i) {
    	std::vector<std::pair<Dtype, int> > sim_vector;
      	for (int j = 0; j < num_1; ++j) {
		  Dtype sim = Similarity(label, i, j, label_dim, bottom[2]->num());
		  sim_vector.push_back(std::make_pair(sim, j));
      	}
      	std::sort(sim_vector.begin(), sim_vector.end(), std::greater<std::pair<Dtype, int> >());
      	for (int k = 0; k < sim_vector.size(); k++) {
			for(int l=k+1; l < sim_vector.size(); l++){
	  			if(sim_vector[k].first == sim_vector[l].first) {
	    		  continue;
				}
	  			++ n_tri;
	  			int a = sim_vector[k].second;
	  			int b = sim_vector[l].second;
	  			Dtype norm1 = 0, norm2 = 0;
	  			for(int j = 0; j < dim; ++j){
	    		  norm1 += pow((bottom_data_0[i*dim + j] - bottom_data_1[a*dim + j]),2);
	    		  norm2 += pow((bottom_data_0[i*dim + j] - bottom_data_1[b*dim + j]),2);
	  			}
	  			if(margin + norm1 - norm2 > 0){
	    		  loss += (margin + norm1 - norm2);
	    		  for(int j = 0; j < dim; ++j){
	      		    diff_0[i*dim +j] += 2*(bottom_data_1[b*dim + j] - bottom_data_1[a*dim + j]);
	      			diff_1[a*dim +j] += 2*(bottom_data_1[a*dim + j] - bottom_data_0[i*dim + j]);
	      			diff_1[b*dim +j] += 2*(bottom_data_0[i*dim + j] - bottom_data_1[b*dim + j]);
	  			  }
				}
				else {
	    		  ++ acc;
	  			}
			}
      	}
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if(n_tri > 0) {
      	top[0]->mutable_cpu_data()[0] = loss / n_tri;
      	if (phase_ == "thief") {
          caffe_scal(bottom[0]->count(), Dtype(0), diff_0);
          caffe_scal(bottom[1]->count(), Dtype(0), diff_1);
        }
        else {
          caffe_scal(bottom[0]->count(), loss_weight / n_tri / margin, diff_0);
          caffe_scal(bottom[1]->count(), loss_weight / n_tri / margin, diff_1);
        }
    }
    else {
      top[0]->mutable_cpu_data()[0] = 0;
    }
    top[1]->mutable_cpu_data()[0] = acc / n_tri;
  }

  template <typename Dtype>
  void CrossModalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  }

#ifdef CPU_ONLY
  STUB_GPU(CrossModalLossLayer);
#endif

  INSTANTIATE_CLASS(CrossModalLossLayer);
  REGISTER_LAYER_CLASS(CrossModalLoss);

}  // namespace caffe
