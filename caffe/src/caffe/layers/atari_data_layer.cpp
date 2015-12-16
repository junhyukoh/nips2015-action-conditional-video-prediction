//#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp"
#include <iostream>
#include <fstream>
#include "caffe/data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/net.hpp"

using namespace cv;
using namespace boost::filesystem;

namespace caffe {

inline Mat LoadImg(const string& data_path, int dir_idx, int img_idx, int channels) {
  char buf[100];
  snprintf(buf, sizeof(buf), "%s/%04u/%05u.png", 
    data_path.c_str(), dir_idx, img_idx);
  Mat img;
  if (channels == 1) {
    img = imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
  }
  else if (channels > 1) {
    img = imread(buf, CV_LOAD_IMAGE_COLOR);
  }
  return img;
}

template <typename Dtype>
void AtariDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_act_ = this->layer_param_.atari_data_param().num_act();
  num_frame_ = this->layer_param_.atari_data_param().num_frame();
  channels_ = this->layer_param_.atari_data_param().channels();
  output_clip_ = this->layer_param_.atari_data_param().out_clip();
  batch_size_ = this->layer_param_.data_param().batch_size();
  streaming_ = this->layer_param_.atari_data_param().streaming();
  load_to_mem_ = this->layer_param_.atari_data_param().load_to_memory();
  output_act_ = this->layer_param_.atari_data_param().out_act();

  int num_out_blobs = 1 + output_act_ + output_clip_;
  CHECK_EQ(num_out_blobs, top.size());
  int output_blob_idx = 1;
  if (output_act_) {
    act_idx_ = output_blob_idx++;
  }
  if (output_clip_) {
    clip_idx_ = output_blob_idx++;
  }
  
  this->data_transformer_.reset(
      new DataTransformer<Dtype>(this->transform_param_, this->phase_));
  this->data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
 
  CHECK(this->layer_param_.transform_param().has_mean_file());
  const string& mean_file = this->layer_param_.transform_param().mean_file();
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  Blob<Dtype> data_mean;
  data_mean.FromProto(blob_proto);
  LOG(INFO) << "Mean : " << data_mean.num() << ","
    << data_mean.channels() << "," 
    << data_mean.width() << ","
    << data_mean.height();
  CHECK_EQ(data_mean.channels(), channels_) 
    << "The number of channels of the mean file"
    << " does not match that of the data.";
  DLOG(INFO) << "Initializing prefetch";
  this->StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void AtariDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LoadData();

  NetParameter net_param;
  Mat image;
  if (load_to_mem_) {
    CHECK_GE(imgs_.size(), 1) << "Images are not found";
    CHECK_GE(imgs_[0].size(), 1) << "Images are not found";
    image = imgs_[0].at(0);
  }
  else {
    image = LoadImg(this->layer_param_.data_param().source(), 0, 0, channels_);
  }

  this->height_ = image.size().height;
  this->width_ = image.size().width;
  this->size_ = channels_ * this->height_ * this->width_;

  Reshape(bottom, top);
  LOG(INFO) << "output data size: " << top[0]->shape_string();

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (output_act_) {
    prefetch_act_.mutable_cpu_data();
  }
  if (output_clip_) {
    prefetch_clip_.mutable_cpu_data();
  }
}

template <typename Dtype>
void AtariDataLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // image
  vector<int> top_shape;
  top_shape.push_back(streaming_ ? num_frame_ : batch_size_);
  top_shape.push_back(streaming_ ? batch_size_ : num_frame_);
  top_shape.push_back(this->channels_);
  top_shape.push_back(this->height_);
  top_shape.push_back(this->width_);
  top[0]->Reshape(top_shape);
  this->prefetch_data_.Reshape(top_shape);

  // action
  if (output_act_) {
    vector<int> act_shape;
    act_shape.push_back(streaming_ ? num_frame_ - 1 : batch_size_);
    act_shape.push_back(streaming_ ? batch_size_ : num_frame_ - 1);
    act_shape.push_back(num_act_);
    top[act_idx_]->Reshape(act_shape);
    prefetch_act_.Reshape(act_shape);
  }

  // clip
  if (this->output_clip_) {
    vector<int> clip_shape;
    clip_shape.push_back(streaming_ ? num_frame_ : batch_size_);
    clip_shape.push_back(streaming_ ? batch_size_ : num_frame_);
    top[clip_idx_]->Reshape(clip_shape);
    this->prefetch_clip_.Reshape(clip_shape);
  }
}

template <typename Dtype>
void AtariDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->WaitForInternalThreadToExit();
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  if (output_act_) {
    caffe_copy(prefetch_act_.count(), prefetch_act_.cpu_data(),
               top[act_idx_]->mutable_cpu_data());
  }
  if (output_clip_) {
    caffe_copy(prefetch_clip_.count(), prefetch_clip_.cpu_data(),
               top[clip_idx_]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  this->StartInternalThread();
}

template <typename Dtype>
void AtariDataLayer<Dtype>::SampleBatch() {
  sample_batch_.clear();
  for (int item_id = 0; item_id < batch_size_; ++item_id) {  
    int idx = caffe_rng_rand() % this->batch_idx_.size();
    sample_batch_.push_back(BatchIdx(this->batch_idx_[idx]));
  }
}

template <typename Dtype>
void AtariDataLayer<Dtype>::InternalThreadEntry() {
  const string& data_path = this->layer_param_.data_param().source();

  // Sample mini-batch indices
  SampleBatch();

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_act = NULL;
  if (output_act_) {
    top_act = this->prefetch_act_.mutable_cpu_data();
    caffe_set(this->prefetch_act_.count(), (Dtype)0., top_act);
  }
  Dtype* top_clip = NULL;
  if (output_clip_) {
    top_clip = this->prefetch_clip_.mutable_cpu_data();
    caffe_set(prefetch_clip_.count(), (Dtype)1., top_clip);
    if (streaming_) {
      caffe_set(prefetch_clip_.count(1), (Dtype)0., top_clip);
    }
    else {
      for (int i = 0; i < batch_size_; ++i) {
        top_clip[prefetch_clip_.offset(i, 0)] = (Dtype)0.;
      }
    }
  }

  Blob<Dtype> uni_blob(1, channels_, height_, width_);
  CHECK_EQ(sample_batch_.size(), batch_size_);
  Mat img;
  for (int item_id = 0; item_id < batch_size_; ++item_id) {  
    vector<int> offset (this->prefetch_data_.num_axes(), 0);

    int dir_idx = sample_batch_[item_id].dir_;
    int img_idx = sample_batch_[item_id].img_;

    if (load_to_mem_) {
      CHECK_GT(imgs_.size(), dir_idx) << "Directory idx is out of range!";
      CHECK_GT(imgs_[dir_idx].size(), img_idx) << "Img idx is out of range!";
    }

    // data
    for (int j = 0; j < num_frame_; ++j) {
      int file_idx = img_idx + j;
      if (load_to_mem_) {
        CHECK_GT(imgs_[dir_idx].size(), file_idx) << "Img idx is out of range!";
      }
      offset[0] = streaming_ ? j : item_id;
      offset[1] = streaming_ ? item_id : j;
      uni_blob.set_cpu_data(top_data + this->prefetch_data_.offset(offset));
      if (load_to_mem_) {
        this->data_transformer_->Transform(imgs_[dir_idx].at(file_idx),
          &uni_blob);
      }
      else {
        img = LoadImg(data_path, dir_idx, file_idx, channels_);
        this->data_transformer_->Transform(img,
          &uni_blob);
      }

      // action
      if (output_act_ && j < num_frame_ - 1) {
        int act_idx = img_idx + j;
        CHECK_LT(act_idx, acts_[dir_idx].size()) << "Action index is out of range!";
        CHECK_LT(acts_[dir_idx].at(act_idx), num_act_) 
          << "Action idx is out of range";
        top_act[prefetch_act_.offset(streaming_ ? j : item_id, 
          streaming_ ? item_id : j) + acts_[dir_idx].at(act_idx)] = 1;
      }
    }

    /*
    if (this->phase_ == TRAIN) {
      if (channels_ > 1) {
        Mat input(this->height_, this->width_*num_frame_, CV_8UC3);
        Blob<Dtype>& mean = this->data_transformer_->mean();
        for (int k = 0; k < this->num_frame_; ++k) {
          for (int h = 0; h < this->height_; ++h) {
            for (int w = 0; w < this->width_; ++w) {
              for (int c = 0; c < this->channels_; ++c) {
                offset[0] = streaming_ ? k : item_id;
                offset[1] = streaming_ ? item_id : k;
                offset[2] = c;
                offset[3] = h;
                offset[4] = w;
                input.data[h*this->width_*num_frame_*channels_ 
                    + k*this->width_*channels_ + w*channels_ + c] = 
                        this->prefetch_data_.data_at(offset) 
                        / this->layer_param().transform_param().scale()
                        + mean.data_at(0, c, h + h_off, w + w_off);
              }
            }
          }
        }

        Mat output(this->height_, this->width_*(overlap_ + step_), CV_8UC3);
        for (int k = 0; k < overlap_ + step_; ++k) {
          for (int h = 0; h < this->height_; ++h) {
            for (int w = 0; w < this->width_; ++w) {
              for (int c = 0; c < this->channels_; ++c) {
                offset[0] = streaming_ ? k : item_id;
                offset[1] = streaming_ ? item_id : k;
                offset[2] = c;
                offset[3] = h;
                offset[4] = w;
                output.data[h*this->width_*(overlap_ + step_)*channels_ 
                  + k*this->width_*channels_ + w*channels_ + c] = 
                      this->prefetch_label_.data_at(offset) 
                      / this->layer_param().transform_param().scale()
                      + mean.data_at(0, c, h + h_off, w + w_off);
              }
            }
          }
        }
        imshow("Window", input);
        imshow("Window2", output);
        waitKey(0);
      }
      else {
        Mat input(this->height_, this->width_*num_frame_, CV_8UC1);
        Blob<Dtype>& mean = this->data_transformer_->mean();
        for (int k = 0; k < this->num_frame_; ++k) {
          for (int h = 0; h < this->height_; ++h) {
            for (int w = 0; w < this->width_; ++w) {
              offset[0] = streaming_ ? k : item_id;
              offset[1] = streaming_ ? item_id : k;
              offset[2] = 0;
              offset[3] = h;
              offset[4] = w;
              input.data[h*this->width_*num_frame_ + k*this->width_ + w] = 
                this->prefetch_data_.data_at(offset) 
                / this->layer_param().transform_param().scale()
                + mean.data_at(0, 0, h + h_off, w + w_off);
            }
          }
        }

        Mat output(this->height_, this->width_*(overlap_ + step_), CV_8UC1);
        for (int k = 0; k < overlap_ + step_; ++k) {
          for (int h = 0; h < this->height_; ++h) {
            for (int w = 0; w < this->width_; ++w) {
              offset[0] = streaming_ ? k : item_id;
              offset[1] = streaming_ ? item_id : k;
              offset[2] = 0;
              offset[3] = h;
              offset[4] = w;
              output.data[h*this->width_*(overlap_ + step_) + k*this->width_ + w] = 
                this->prefetch_label_.data_at(offset) 
                / this->layer_param().transform_param().scale()
                + mean.data_at(0, 0, h + h_off, w + w_off);
            }
          }
        }
        imshow("Window", input);
        imshow("Window2", output);
        waitKey(0);
      }
    }
    */
  }
}

template <typename Dtype>
void AtariDataLayer<Dtype>::LoadData() {
  string data_path = this->layer_param_.data_param().source();
  LOG(INFO) << "Opening DB : " << data_path;
  CHECK_EQ(is_directory(data_path), true) 
    << "Cannot Open Source : " << data_path;

  char buf[100];
  std::string file_path, dir_path;

  int img_idx = 0;
  int dir_idx = 0;
  int total_img = 0;
  int total_episode = 0;
  while (true) {
    std::vector<Mat> imgs;
    std::vector<int> acts;
    std::vector<int> rewards;
    std::vector<std::vector<Dtype> > q_values;
    snprintf(buf, sizeof(buf), "%s/%04u", data_path.c_str(), dir_idx);
    dir_path = buf;

    if (!is_directory(dir_path)) {
      break; 
    }
    img_idx = 0;
    // Load images
    while (true) {
      if (load_to_mem_) {
        snprintf(buf, sizeof(buf), "/%05u.png", (unsigned int)imgs.size());
      }
      else {
        snprintf(buf, sizeof(buf), "/%05u.png", img_idx);
      }
      file_path = dir_path + buf;
      if (!exists(file_path)) {
        break;
      }
      if (load_to_mem_) {
        Mat img;
        if (channels_ == 1) {
          img = imread(file_path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
          CHECK_EQ(CV_8UC1, img.type());
        }
        else if (channels_ > 1) {
          img = imread(file_path.c_str(), CV_LOAD_IMAGE_COLOR);
          CHECK_EQ(CV_8UC3, img.type());
        }
        imgs.push_back(img);
      }
      total_img++;
      img_idx++;
    }

    if (load_to_mem_) {
      CHECK_GE(imgs.size(), num_frame_) << dir_path << " has little images.";
    }
    for (int i = 0; i <= img_idx - num_frame_; ++i) {
      batch_idx_.push_back(BatchIdx(total_episode, i));
    }

    if (load_to_mem_) {
      this->imgs_.push_back(imgs);
    }

    // Load actions
    if (output_act_) {
      std::string act_path(dir_path + "/act.log");
      std::string line;
      std::ifstream file(act_path.c_str());
      CHECK_EQ(file.is_open(), true) << "Action file not found : " << act_path;
      while (getline(file, line)) {
        acts.push_back(atoi(line.c_str()));
      }
      file.close();
      this->acts_.push_back(acts);
    }
    total_episode++;
    
    LOG(INFO) << dir_path << " is loaded. (" 
      << total_episode << ", " << total_img << ")";
    dir_idx++;
  }

  LOG(INFO) << total_episode << " episodes, " 
    << total_img << " images are loaded";
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(AtariDataLayer, Forward);
#endif

INSTANTIATE_CLASS(AtariDataLayer);
REGISTER_LAYER_CLASS(AtariData);
}
