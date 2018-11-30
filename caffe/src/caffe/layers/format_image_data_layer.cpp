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
FormatImageDataLayer<Dtype>::~FormatImageDataLayer<Dtype>() {
  this->StopInternalThread();
}
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}
std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

template <typename Dtype>
void FormatImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.format_image_data_param().new_height();
  const int new_width  = this->layer_param_.format_image_data_param().new_width();
  const bool is_color  = this->layer_param_.format_image_data_param().is_color();
  string root_folder = this->layer_param_.format_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.format_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  //int label;
  //while (infile >> filename >> label) {
    //lines_.push_back(std::make_pair(filename, label));
  //}
  int label_num = 0;
  for(std::string line; std::getline(infile, line);)
  {
	  string filename;
	  //vector<int> labels; ////////////////// modify
	  vector<double> labels;
	  std::vector<std::string> x = split(line,' ');
	  int n = x.size();
	  filename = x[0];
	  for (int i=1; i<n; i++)
	  {
	    //labels.push_back(atoi(x[i].c_str()));////////////// modify
	    labels.push_back(atof(x[i].c_str()));
	  }
	  if(label_num < labels.size())
		  label_num = labels.size();
	  lines_.push_back(make_pair(filename, labels));
  }
  //added by ysm
  label_num_ = label_num;
  //added by ysm end

  if (this->layer_param_.format_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.format_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.format_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.format_image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(1, batch_size);
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(label_num);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void FormatImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  //added by ysm
  //int label_size = lines_.size()/label_num_;
  //for(int i = 0; i < label_num_; ++i) {
  //    shuffle(lines_.begin() + i * label_size, lines_.begin() + (i+1) * label_size, prefetch_rng);
  //}
  //added by ysm end
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FormatImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  FormatImageDataParameter format_image_data_param = this->layer_param_.format_image_data_param();
  const int batch_size = format_image_data_param.batch_size();
  const int new_height = format_image_data_param.new_height();
  const int new_width = format_image_data_param.new_width();
  const bool is_color = format_image_data_param.is_color();
  string root_folder = format_image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  //added by ysm 
  //const int label_size_each_batch = batch_size/label_num_;
  //const int label_size = lines_size / label_num_;
  //added by ysm end
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
	//
	//added by ysm
	//int current_label = item_id / label_size_each_batch;
	//int current_batch = lines_id_ / batch_size;
	//int current_lines_id = current_label * label_size + current_batch * label_size_each_batch + item_id % label_size_each_batch;
    //CHECK_GT(lines_size, current_lines_id) << current_label << " " << current_batch << " " << item_id << " " << label_size << " " << label_size_each_batch;
	//added by ysm end
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    //cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[current_lines_id].first,
        //new_height, new_width, is_color);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    //CHECK(cv_img.data) << "Could not load " << lines_[current_lines_id].first;
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    //prefetch_label[item_id] = lines_[lines_id_].second;
	//vector<int> labels = lines_[current_lines_id].second;
        vector<double> labels = lines_[lines_id_].second;
        //vector<int> labels = lines_[lines_id_].second; /////////////////////modify
	int label_num = labels.size();
	for (int k = 0; k < label_num; ++k){
		prefetch_label[item_id*label_num+k] = labels[k];
	}
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.format_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(FormatImageDataLayer);
REGISTER_LAYER_CLASS(FormatImageData);

}  // namespace caffe
#endif  // USE_OPENCV
