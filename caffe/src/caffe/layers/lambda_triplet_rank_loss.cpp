#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


  template <typename Dtype>
  void LambdaTripletRankLossLayer<Dtype>::Reshape(
						const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
    topk_ = this->layer_param_.triplet_param().topk();
    int num = bottom[0]->num();
    topk_ = num-1 > topk_ ? topk_ : num-1;
    type_ =  this->layer_param_.triplet_param().type();
    margin_ = this->layer_param_.triplet_param().margin();    
  }


  template <typename Dtype>
  int LambdaTripletRankLossLayer<Dtype>::LabelSimilarity(
							 const Dtype* labels, const int i, const int j, const int dim ) {
    int s_sim = 0;
    for(int k=0; k < dim; k++){
      if(labels[i*dim + k] > 0 && labels[i*dim + k] == labels[j*dim + k]){
	s_sim++;
      }
    }
    return s_sim;
  }

  template <typename Dtype>
  Dtype LambdaTripletRankLossLayer<Dtype>::HammingSimilarity(
							   const Dtype* bottom_data, const int i, const int j, const int dim ) {
    Dtype s_sim = 0;
    for(int k=0; k < dim; k++){
      if((bottom_data[i*dim + k] > 0.5 && bottom_data[j*dim + k] > 0.5) || (bottom_data[i*dim + k] <= 0.5 && bottom_data[j*dim +k] <= 0.5)  ){
	s_sim++;
      }
    }
    return s_sim;
  }


  template <typename Dtype>
  Dtype LambdaTripletRankLossLayer<Dtype>::weightTriplet(const vector<std::pair<Dtype, vector<int> > > hash_sim_vector,
						       const int k, const int l, const Dtype best_dcg ) {
    if(k > topk_ && l > topk_)
      return 0.0;

    Dtype changes;
    vector<int> tmp_k = hash_sim_vector[k].second;
    vector<int> tmp_l = hash_sim_vector[l].second;
     
    if(type_ == "NDCG"){
      Dtype ch = ((1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1])));
      changes = ch / best_dcg;
    } else if(type_ == "DCG"){
      changes = ((1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1])));
    } else{ 
      changes = (tmp_k[1] -  tmp_l[1]);
    }
    if(changes < 0)
      changes = -changes;
    return changes;
  }

  template <typename Dtype>
  void LambdaTripletRankLossLayer<Dtype>::Forward_cpu(
						    const vector<Blob<Dtype>*>& bottom,
						    const  vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int label_dim = bottom[1]->count() / bottom[1]->num();
    Dtype loss=0;
    Dtype  n_tri = 0;
       
    for (int i = 0; i < num; ++i) {
      std::vector<std::pair<Dtype, int> > label_sim_vector;
      std::vector<std::pair<Dtype, vector<int> > > hash_sim_vector;
      for (int j = 0; j < num; ++j) {
	if(j == i)
	  continue;
	int sim = LabelSimilarity(label, i, j,  label_dim );
	Dtype hash_sim = HammingSimilarity(bottom_data,i,j, dim);
	vector<int> tmp;
	tmp.push_back(j);
	tmp.push_back(sim);
	hash_sim_vector.push_back(std::make_pair(hash_sim,tmp));
	label_sim_vector.push_back(std::make_pair(sim, j));
      }
      Dtype best_dcg = 0.0;
      if(type_ == "NDCG"){ 
	std::sort(label_sim_vector.begin(), label_sim_vector.end(), std::greater<std::pair<Dtype, int> >());
	for(int j=0; j < topk_; j++)
	  best_dcg += (pow(2.0, label_sim_vector[j].first)-1.0)/log(j+2);
      }
      std::sort(hash_sim_vector.begin(), hash_sim_vector.end(), std::greater<std::pair<Dtype, vector<int> > >());
      for (int k = 0; k < hash_sim_vector.size(); k++) {
	vector<int> tmp_k = hash_sim_vector[k].second;
        for(int l=k+1; l < hash_sim_vector.size(); l++){
	  vector<int> tmp_l = hash_sim_vector[l].second;
          if(tmp_k[1] == tmp_l[1])
	    continue;
	  n_tri++;
        
	  Dtype changes = 1; //weightTriplet(hash_sim_vector, k,  l,  best_dcg );
	  if(type_ == "NDCG"){
	    Dtype ch = (1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1]));
	    changes = ch / best_dcg;
	  } else if(type_ == "DCG"){
	    changes = (1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1]));
	  } else if(type_ == "ABS"){ 
	    changes = (tmp_k[1] -  tmp_l[1]);
	  } else{
	    changes = pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1]);
	  }
	  if(changes < 0)
	    changes = -changes;

	  int a, b;
	  if(tmp_k[1] > tmp_l[1]){
	    a = tmp_k[0];
	    b = tmp_l[0];
	  }else{
	    a = tmp_l[0];
	    b = tmp_k[0];
	  }
	
	  Dtype norm1=0, norm2 = 0;
	  for(int j=0; j < dim; ++j){
	    norm1 += pow((bottom_data[i*dim + j] - bottom_data[a*dim + j]),2);
	    norm2 += pow((bottom_data[i*dim + j] - bottom_data[b*dim + j]),2);
	  }
	  if(margin_ +norm1 - norm2 > 0){
	    loss += 1; // (margin_ + norm1 - norm2)*changes;
	  }
	}
      }
    }
    if(n_tri > 0)
      top[0]->mutable_cpu_data()[0] = (n_tri-loss)/n_tri;
    else
      top[0]->mutable_cpu_data()[0] = 0;
    //return loss/num;
  }

  template <typename Dtype>
  void LambdaTripletRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
						     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    memset(diff, 0, bottom[0]->count()*sizeof(Dtype));
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int label_dim = bottom[1]->count() / bottom[1]->num();
  
    Dtype  n_tri = 0;

    for (int i = 0; i < num; ++i) {   
      std::vector<std::pair<Dtype, int> > label_sim_vector;
      std::vector<std::pair<Dtype, vector<int> > > hash_sim_vector;
      for (int j = 0; j < num; ++j) {
	if(j == i)
	  continue;
	int sim = LabelSimilarity(label, i, j,  label_dim );
	Dtype hash_sim = HammingSimilarity(bottom_data,i,j, dim);
	vector<int> tmp;
	tmp.push_back(j);
	tmp.push_back(sim);
	hash_sim_vector.push_back(std::make_pair(hash_sim,tmp));
	label_sim_vector.push_back(std::make_pair(sim, j));
      }
      Dtype best_dcg = 0.0;
      if(type_ == "NDCG"){ 
	std::sort(label_sim_vector.begin(), label_sim_vector.end(), std::greater<std::pair<Dtype, int> >());
	for(int j=0; j < topk_; j++)
	  best_dcg += (pow(2.0, label_sim_vector[j].first)-1.0)/log(j+2);
      }
      std::sort(hash_sim_vector.begin(), hash_sim_vector.end(), std::greater<std::pair<Dtype, vector<int> > >());
      for (int k = 0; k < hash_sim_vector.size(); k++) {
	vector<int> tmp_k = hash_sim_vector[k].second;
        for(int l=k+1; l < hash_sim_vector.size(); l++){
	  vector<int> tmp_l = hash_sim_vector[l].second;
          if(tmp_k[1] == tmp_l[1])
	    continue;
	  n_tri++;
        
	  Dtype changes = 1; //weightTriplet(hash_sim_vector, k,  l,  best_dcg );
	  if(type_ == "NDCG"){
	    Dtype ch = (1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1]));
	    changes = ch / best_dcg;
	  } else if(type_ == "DCG"){
	    changes = (1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1]));
	  } else if(type_ == "ABS"){ 
	    changes = (tmp_k[1] -  tmp_l[1]);
	  } else{
	    changes = pow(2.0,tmp_k[1]) - pow(2.0, tmp_l[1]);
	  }
	  if(changes < 0)
	    changes = -changes;

	  int a, b;
	  if(tmp_k[1] > tmp_l[1]){
	    a = tmp_k[0];
	    b = tmp_l[0];
	  }else{
	    a = tmp_l[0];
	    b = tmp_k[0];
	  }
	
	  Dtype norm1=0, norm2 = 0;
	  for(int j=0; j < dim; ++j){
	    norm1 += pow((bottom_data[i*dim + j] - bottom_data[a*dim + j]),2);
	    norm2 += pow((bottom_data[i*dim + j] - bottom_data[b*dim + j]),2);
	  }
	  if(margin_ +norm1 - norm2 > 0){
	    for(int j=0; j < dim; ++j){
	      diff[i*dim +j] += 2*changes*(bottom_data[b*dim + j] - bottom_data[a*dim + j]);
	      diff[a*dim +j] += 2*changes*(bottom_data[a*dim + j] - bottom_data[i*dim + j]);
	      diff[b*dim +j] += 2*changes*(bottom_data[i*dim + j] - bottom_data[b*dim + j]); 
	    }
	  }
	} 
      }
    }   
    // Scale down gradient
    if(n_tri > 0)
      caffe_scal(bottom[0]->count(), Dtype(1) / n_tri / margin_, diff);
    /*
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
      std::vector<std::pair<Dtype, vector<int> > > hash_sim_vector;
      for (int j = 0; j < num; ++j) {
      if(j == i)
      continue;
      Dtype sim = 0;
      for(int k=0; k < label_dim; k++){
      if(label[i*label_dim + k] > 0 && label[i*label_dim + k] == label[j*label_dim + k]){
      sim++;
      }
      }
      sim_vector.push_back(std::make_pair(sim, j));

      Dtype hash_sim = 0;
      for(int k=0; k < dim; k++){
      if((bottom_data[i*dim + k] > 0.5 && bottom_data[j*dim + k] > 0.5) || (bottom_data[i*dim + k] <= 0.5 && bottom_data[j*dim +k] <= 0.5)  ){
      hash_sim++;
      }
      }
      vector<int> tmp;
      tmp.push_back(j);
      tmp.push_back(sim);
      hash_sim_vector.push_back(std::make_pair(hash_sim,tmp));
      }
     
      std::sort(hash_sim_vector.begin(), hash_sim_vector.end(), std::greater<std::pair<Dtype, vector<int> > >());

      Dtype best_dcg = 0.0;
      if(type_ == "NDCG"){ 
      std::sort(sim_vector.begin(), sim_vector.end(), std::greater<std::pair<Dtype, int> >());
      for(int j=0; j < topk_; j++)
      best_dcg += (pow(2.0, sim_vector[j].first)-1.0)/log(j+2);
      }
     
      for (int k = 0; k < hash_sim_vector.size(); k++) {
      for(int l=k+1; l < hash_sim_vector.size(); l++){
      if(hash_sim_vector[k].second[1] == hash_sim_vector[l].second[1])
      continue;
      n_tri++;

      Dtype changes;

      if(type_ == "NDCG"){
      Dtype ch = abs((1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,hash_sim_vector[k].second[1]) - pow(2.0, hash_sim_vector[l].second[1])));
      changes = ch / best_dcg;
      } else if(type_ == "DCG"){
      changes = abs((1.0/log(k+2) - 1.0/log(l+2)) * (pow(2.0,hash_sim_vector[k].second[1]) - pow(2.0, hash_sim_vector[l].second[1])));
      } else{ 
      changes = abs(hash_sim_vector[k].second[1] -  hash_sim_vector[l].second[1]);
      }

      int a = hash_sim_vector[k].second[0];
      int b = hash_sim_vector[l].second[0];

      Dtype norm1=0, norm2 = 0;
      for(int j=0; j < dim; ++j){
      norm1 += pow((bottom_data[i*dim + j] - bottom_data[a*dim + j]),2);
      norm2 += pow((bottom_data[i*dim + j] - bottom_data[b*dim + j]),2);
      }
      if(margin +norm1 - norm2 > 0){
      for(int j=0; j < dim; ++j){
      diff[i*dim +j] += 2*changes*(bottom_data[b*dim + j] - bottom_data[a*dim + j]);
      diff[a*dim +j] += 2*changes*(bottom_data[a*dim + j] - bottom_data[i*dim + j]);
      diff[b*dim +j] += 2*changes*(bottom_data[i*dim + j] - bottom_data[b*dim + j]); 
      }
      }
      }
      } 
      }     
      // Scale down gradient
      if(n_tri > 0)
      caffe_scal(bottom[0]->count(), Dtype(1) / n_tri / margin, diff);
    */
  }

#ifdef CPU_ONLY
  STUB_GPU(LambdaTripletRankLossLayer);
#endif

  INSTANTIATE_CLASS(LambdaTripletRankLossLayer);
  REGISTER_LAYER_CLASS(LambdaTripletRankLoss);

}  // namespace caffe
