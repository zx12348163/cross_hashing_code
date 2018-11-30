#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
int PairFormatImageDataLayer<Dtype>::Similarity(const vector<int>& labelx, const vector<int>& labely) {
  CHECK(labelx.size() == labely.size());
  for(int i = 0; i < labelx.size(); ++i) {
	  if(1 == labelx[i] && labelx[i] == labely[i])
		  return 1;
  }
  return 0;
}


template <typename Dtype>
PairFormatImageDataLayer<Dtype>::~PairFormatImageDataLayer<Dtype>() {
  this->StopInternalThread();
}
//std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
//	std::stringstream ss(s);
//	std::string item;
//	while (std::getline(ss, item, delim)) {
//		elems.push_back(item);
//	}
//	return elems;
//}
std::vector<std::string> split(const std::string &s, char delim);
//	std::vector<std::string> elems;
//	split(s, delim, elems);
//	return elems;
//}

template <typename Dtype>
void PairFormatImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.pair_format_image_data_param().new_height();
  const int new_width  = this->layer_param_.pair_format_image_data_param().new_width();
  const bool is_color  = this->layer_param_.pair_format_image_data_param().is_color();
  string root_folder = this->layer_param_.pair_format_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.pair_format_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label_num = 0;
  for(std::string line; std::getline(infile, line);)
  {
	  string filename;
	  vector<int> labels;
	  std::vector<std::string> x = split(line,' ');
	  int n = x.size();
	  filename = x[0];
	  for (int i=1; i<n; i++)
	  {
		  labels.push_back(atoi(x[i].c_str()));
	  }
	  if(label_num < labels.size())
		  label_num = labels.size();
	  lines_.push_back(make_pair(filename, labels));						
  }

  //if (this->layer_param_.pair_format_image_data_param().shuffle()) {
    // randomly shuffle data
    //LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    //ShuffleImages();
  //}
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  for(int i = 0; i < lines_.size() - 1; ++i) {
	  for(int j = i + 1; j < lines_.size(); ++j) {
		if(Similarity(lines_[i].second, lines_[j].second)) {
			similar_pairs_.push_back(std::make_pair(i,j));
		}
		else {
			dissimilar_pairs_.push_back(std::make_pair(i,j));
		}
	  }
  }
  similar_ind_ = 0;
  dissimilar_ind_ = 0;
  ShuffleSimilarPairs();
  ShuffleDissimilarPairs();
  LOG(INFO) << "A total of " << similar_pairs_.size() << " similar pairs.";
  LOG(INFO) << "A total of " << dissimilar_pairs_.size() << " dissimilar pairs.";

  //lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  //if (this->layer_param_.pair_format_image_data_param().rand_skip()) {
  //  unsigned int skip = caffe_rng_rand() %
  //      this->layer_param_.pair_format_image_data_param().rand_skip();
  //  LOG(INFO) << "Skipping first " << skip << " data points.";
  //  CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
  //  lines_id_ = skip;
  //}
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[0].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[0].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.pair_format_image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size * 2;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(1, batch_size);
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void PairFormatImageDataLayer<Dtype>::ShuffleSimilarPairs() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(similar_pairs_.begin(), similar_pairs_.end(), prefetch_rng);
}
template <typename Dtype>
void PairFormatImageDataLayer<Dtype>::ShuffleDissimilarPairs() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(dissimilar_pairs_.begin(), dissimilar_pairs_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void PairFormatImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  PairFormatImageDataParameter pair_format_image_data_param = this->layer_param_.pair_format_image_data_param();
  const int batch_size = pair_format_image_data_param.batch_size();
  const int new_height = pair_format_image_data_param.new_height();
  const int new_width = pair_format_image_data_param.new_width();
  const bool is_color = pair_format_image_data_param.is_color();
  string root_folder = pair_format_image_data_param.root_folder();
  const float ratio = pair_format_image_data_param.ratio();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[0].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[0].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size * 2;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  std::vector< std::pair< std::pair< int, int >, int > > pair_batch;
  for(int i = 0; i < static_cast<int>(batch_size * ratio); ++i) {
	  pair_batch.push_back(std::make_pair(similar_pairs_[similar_ind_], 1));
	  ++similar_ind_;
	  if(similar_ind_ >= similar_pairs_.size()) {
		  similar_ind_ = 0;
		  ShuffleSimilarPairs();
	  }
  }
  for(int i = 0; i < batch_size - static_cast<int>(batch_size * ratio); ++i) {
	  pair_batch.push_back(std::make_pair(dissimilar_pairs_[dissimilar_ind_], 0));
	  ++dissimilar_ind_;
	  if(dissimilar_ind_ >= dissimilar_pairs_.size()) {
		  dissimilar_ind_ = 0;
		  ShuffleDissimilarPairs();
	  }
  }
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(pair_batch.begin(), pair_batch.end(), prefetch_rng);

  // datum scales
  //const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
	// first image in pair
    timer.Start();
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[pair_batch[item_id].first.first].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[pair_batch[item_id].first.first].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id * 2);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

	//second image in pair
    timer.Start();
    cv_img = ReadImageToCVMat(root_folder + lines_[pair_batch[item_id].first.second].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[pair_batch[item_id].first.second].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    offset = batch->data_.offset(item_id * 2 + 1);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = pair_batch[item_id].second;
	//vector<int> labels = lines_[current_lines_id].second;
	//vector<int> labels = lines_[lines_id_].second;
	//int label_num = labels.size();
	//for (int k = 0; k < label_num; ++k){
	//	prefetch_label[item_id*label_num+k] = labels[k];
	//}
    // go to the next iter
    //lines_id_++;
    //if (lines_id_ >= lines_size) {
    //  // We have reached the end. Restart from the first.
    //  DLOG(INFO) << "Restarting data prefetching from start.";
    //  lines_id_ = 0;
    //  if (this->layer_param_.format_image_data_param().shuffle()) {
    //    ShuffleImages();
    //  }
    //}
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(PairFormatImageDataLayer);
REGISTER_LAYER_CLASS(PairFormatImageData);

}  // namespace caffe
#endif  // USE_OPENCV
