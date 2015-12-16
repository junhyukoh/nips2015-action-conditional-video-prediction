#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void AtariDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->WaitForInternalThreadToExit();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_act_) {
    caffe_copy(this->prefetch_act_.count(), this->prefetch_act_.cpu_data(),
               top[act_idx_]->mutable_gpu_data());
  }
  if (this->output_clip_) {
    caffe_copy(this->prefetch_clip_.count(), this->prefetch_clip_.cpu_data(),
               top[clip_idx_]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->StartInternalThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(AtariDataLayer);

}  // namespace caffe
