#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
// #include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BernoulliLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    unsigned int sample = 0;
    caffe_rng_bernoulli(1, bottom_data[i], &sample);
    top_data[i] = (Dtype)sample;
  }
}

template <typename Dtype>
void BernoulliLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(BernoulliLayer);
#endif

INSTANTIATE_CLASS(BernoulliLayer);
REGISTER_LAYER_CLASS(Bernoulli);

}  // namespace caffe
