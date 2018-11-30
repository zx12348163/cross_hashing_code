// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert text to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    text_to_db [FLAGS] LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/text_to_db");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[1]);
  std::vector<vector<float> > lines;
  std::string line;
  float data;
  char c;
  //getline(infile, line);
  while (getline(infile, line)) {
	std::vector<float> temp;
	std::istringstream ss(line);
	ss >> data;
	temp.push_back(data);
	while(ss >> c >> data) {
		temp.push_back(data);
	}
	//temp.pop_back();
	lines.push_back(temp);
  }
  //std::ifstream infile2(argv[3]);
  //getline(infile2, line);
  //while (getline(infile2, line)) {
  //  std::vector<float> temp;
  //  std::istringstream ss(line);
  //  ss >> data;
  //  while(ss >> c >> data) {
  //  	temp.push_back(data);
  //  }
  //  lines.push_back(temp);
  //}
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " data.";
  LOG(INFO) << "A total of " << lines[0].size() << " dim.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[2], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  Datum datum;
  int count = 0;
  int dim = lines[0].size();
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
	datum.set_channels(dim);
	datum.set_height(1);
	datum.set_width(1);
	datum.clear_float_data();
	for(int i = 0; i < lines[line_id].size(); ++i) {
		datum.add_float_data(lines[line_id][i]);
	}
    //bool status;
    //std::string enc = encode_type;
    //if (encoded && !enc.size()) {
    //  // Guess the encoding type from the file name
    //  string fn = lines[line_id].first;
    //  size_t p = fn.rfind('.');
    //  if ( p == fn.npos )
    //    LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
    //  enc = fn.substr(p);
    //  std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    //}
    //status = ReadImageToDatum(root_folder + lines[line_id].first,
    //    lines[line_id].second, resize_height, resize_width, is_color,
    //    enc, &datum);
    //if (status == false) continue;
    //if (check_size) {
    //  if (!data_size_initialized) {
    //    data_size = datum.channels() * datum.height() * datum.width();
    //    data_size_initialized = true;
    //  } else {
    //    const std::string& data = datum.data();
    //    CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
    //        << data.size();
    //  }
    //}
    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d", line_id);

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
