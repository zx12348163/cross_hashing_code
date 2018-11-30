#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void TripletRankLossLabelLayer<Dtype>::Reshape(
						 const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
    top[1]->Reshape(1, 1, 1, 1);
  }


  template <typename Dtype>
  Dtype TripletRankLossLabelLayer<Dtype>::Similarity(
						     const Dtype* labels, int i, int j, const int dim, const int batch_size) {
    Dtype s_sim = 0;
    i %= batch_size;
    j %= batch_size;
    /***/
    if (i == j) {
        return dim;
    }
    for(int k = 0; k < dim; ++ k){
	    s_sim += labels[i*dim+k] * labels[j*dim+k];
    }
    return s_sim;
    /****/
    //if (abs(j-i) == batch_size) return 2;
    //for(int k = 0; k < dim; ++ k){
    //    if (labels[i*dim+k] * labels[j*dim+k] == 1) return 1;
    //}
    //return 0;
  }


  template <typename Dtype>
  void TripletRankLossLabelLayer<Dtype>::Forward_cpu(
						     const vector<Blob<Dtype>*>& bottom,
						     const  vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    memset(diff, 0, bottom[0]->count()*sizeof(Dtype));
    
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int label_dim = bottom[1]->count() / bottom[1]->num();
    Dtype margin = this->layer_param_.contrastive_loss_param().margin();    
    Dtype loss=0;
    Dtype  n_tri = 0;
    Dtype acc = 0;
    for (int i = 0; i < num ; ++i) {
    	std::vector<std::pair<Dtype, int> > sim_vector;
      	for (int j = 0; j < num; ++j) {
	    	if(j == i) {
	  	  		continue;
			}
		Dtype sim = Similarity(label, i, j, label_dim, bottom[1]->num());
			sim_vector.push_back(std::make_pair(sim, j));
      	}
      	std::sort(sim_vector.begin(), sim_vector.end(), std::greater<std::pair<Dtype, int> >());
      	//Dtype best_dcg = 0.0;
      	//for(int k=0;k<sim_vector.size();k++) {
	//		best_dcg += (pow(2.0, sim_vector[k].first)-1.0)/log(k+2);
	//	}
      	//if(best_dcg == 0) {
	//		continue;
	//	}
      	for (int k = 0; k < sim_vector.size(); k++) {
			for(int l=k+1; l < sim_vector.size(); l++){
	  			if(sim_vector[k].first == sim_vector[l].first) {
	    			continue;
				}
	  			++ n_tri;
	  			Dtype changes = 1;
                //(1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,sim_vector[k].first) - pow(2.0, sim_vector[l].first));
	  			//if(changes < 0) {
	    		//	changes = -changes;
				//}
	  			int a = sim_vector[k].second;
	  			int b = sim_vector[l].second;
	  			Dtype norm1=0, norm2 = 0;
	  			for(int j=0; j < dim; ++j){
	    			norm1 += pow((bottom_data[i*dim + j] - bottom_data[a*dim + j]),2);
	    			norm2 += pow((bottom_data[i*dim + j] - bottom_data[b*dim + j]),2);
	  			}
	  			//LOG(INFO) << norm1 << " " << norm2;
	  			if(margin + norm1 - norm2 > 0){
	    			  loss += (margin + norm1 - norm2)*changes;
	    			  for(int j=0; j < dim; ++j){
	      				diff[i*dim +j] += 2*changes*(bottom_data[b*dim + j] - bottom_data[a*dim + j]);
	      				diff[a*dim +j] += 2*changes*(bottom_data[a*dim + j] - bottom_data[i*dim + j]);
	      				diff[b*dim +j] += 2*changes*(bottom_data[i*dim + j] - bottom_data[b*dim + j]);
	  			  }
				}
				else {
	    			  acc++;
	  			}
			}
      	}
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if(n_tri > 0) {
      	top[0]->mutable_cpu_data()[0] = loss/n_tri;
      	if (phase_ == "thief") caffe_scal(bottom[0]->count(), Dtype(0), diff);
        else caffe_scal(bottom[0]->count(), loss_weight / n_tri / margin, diff);
    }
    else {
      top[0]->mutable_cpu_data()[0] = 0;
    }
    top[1]->mutable_cpu_data()[0] = acc / n_tri;
  }

  template <typename Dtype>
  void TripletRankLossLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
						      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /***************
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    memset(diff, 0, bottom[0]->count()*sizeof(Dtype));
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int label_dim = bottom[1]->count() / bottom[1]->num();

    Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    Dtype  n_tri = 0;
    for (int i = 0; i < num; ++i) {
      	std::vector<std::pair<Dtype, int> > sim_vector;
      	for (int j = 0; j < num; ++j) {
			if(j == i) continue;
			Dtype sim = Similarity(label, i, j,  label_dim, bottom[1]->num());
			sim_vector.push_back(std::make_pair(sim, j));
      	}
      	std::sort(sim_vector.begin(), sim_vector.end(), std::greater<std::pair<Dtype, int> >());
      	Dtype best_dcg = 0.0;
      	for(int k=0;k< sim_vector.size();k++) {
			best_dcg += (pow(2.0, sim_vector[k].first)-1.0)/log(k+2);
		}
      	if(best_dcg == 0) continue;
			for (int k = 0; k < sim_vector.size(); k++) {
				for(int l=k+1; l < sim_vector.size(); ++l){
	  				//if(sim_vector[k].first == sim_vector[l].first) // for hpc hash || sim_vector[l].first > 4)
	  				if(sim_vector[k].first == sim_vector[l].first) continue;	 	 
	  				n_tri++;
	  				Dtype changes = 1;
                    //(1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,sim_vector[k].first) - pow(2.0, sim_vector[l].first));
	  				//if(changes < 0) {
	    			//	changes = -changes;
					//}
	  				int a = sim_vector[k].second;
	  				int b = sim_vector[l].second;

	  				Dtype norm1=0, norm2=0;
	  				for(int j=0; j < dim; ++j){
	    				norm1 += pow((bottom_data[i*dim + j] - bottom_data[a*dim + j]),2);
	    				norm2 += pow((bottom_data[i*dim + j] - bottom_data[b*dim + j]),2);
	  				}
	  				if(margin + norm1 - norm2 > 0){
	    			//if(print_count%100 == 0) std::cout << "norm1:" << norm1 << " norm2:" << norm2 << std::endl;
	    			for(int j=0; j < dim; ++j){
	      				diff[i*dim +j] += 2*changes*(bottom_data[b*dim + j] - bottom_data[a*dim + j]);
	      				diff[a*dim +j] += 2*changes*(bottom_data[a*dim + j] - bottom_data[i*dim + j]);
	      				diff[b*dim +j] += 2*changes*(bottom_data[i*dim + j] - bottom_data[b*dim + j]); 
	      				//if(print_count%100 == 0) std::cout << bottom_data[i*dim + j] << " ";
	      				//diff[i*dim +j] += 2*(bottom_data[b*dim + j] - bottom_data[a*dim + j]);
	      				//diff[a*dim +j] += 2*(bottom_data[a*dim + j] - bottom_data[i*dim + j]);
	      				//diff[b*dim +j] += 2*(bottom_data[i*dim + j] - bottom_data[b*dim + j]); 
	    			}
	    			//if(print_count % 100 == 0) std::cout << std::endl;
	  			}
			}
      	} 
    }       
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if(n_tri > 0) {
      if (phase_ == "thief") caffe_scal(bottom[0]->count(), Dtype(0), diff);
      else caffe_scal(bottom[0]->count(), loss_weight / n_tri / margin, diff);
    }
    *************/
  }

#ifdef CPU_ONLY
  STUB_GPU(TripletRankLossLabelLayer);
#endif

  INSTANTIATE_CLASS(TripletRankLossLabelLayer);
  REGISTER_LAYER_CLASS(TripletRankLossLabel);

}  // namespace caffe
