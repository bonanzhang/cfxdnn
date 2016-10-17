#include "sequential_network.h"
#include <iostream>
SequentialNetwork::SequentialNetwork(size_t batch_size, 
                                     size_t channel, 
                                     size_t height, 
                                     size_t width, 
                                     size_t classes) {
    batch_size_ = batch_size;
    channel_ = channel;
    height_ = height;
    width_ = width;
    classes_ = classes;
}
SequentialNetwork::~SequentialNetwork() {
    for (auto r : data_tensors_) {
        dnnReleaseBuffer_F32(r);
    }
    for (auto r : gradient_tensors_) {
        dnnReleaseBuffer_F32(r);
    }
    for (auto p : net_) {
        delete p;
    }
}
int SequentialNetwork::add_layer(Layer *l) {
    int id = layers_.size();
    layers_.push_back(l);
    return id;
}
void SequentialNetwork::finalize_layers() {
    // the first resource is temporarily empty
    // it will be set by train
    data_tensors_.push_back(nullptr);
    gradient_tensors_.push_back(nullptr);
    vector<size_t> input_dimensions;
    input_dimensions.push_back(width_);
    input_dimensions.push_back(height_);
    input_dimensions.push_back(channel_);
    input_dimensions.push_back(batch_size_);
    for (int i = 0; i < layers_.size(); i++) {
        vector<size_t> output_dimensions;
        // each time a primitive is contructed,
        // it requires the input tensor dimensions
        // it gives back the output tensor dimensions,
        // which is used by the next layer as the input
        net_.push_back(new Primitive(layers_[i], input_dimensions, output_dimensions));
        input_dimensions = output_dimensions;
        // the tensor resources are allocated here
        // the data and the gradient tensors have the same dimensions
        // this is WHCN
        size_t const dim = output_dimensions.size();
        size_t sizes[dim];
        size_t str = 1;
        size_t strides[dim];

  std::cout << "odim " << i << ": ";
  for(int i = 0; i < dim; i++)
    std::cout << output_dimensions[i] << " ";
  std::cout << std::endl;
        for (int i = 0; i < dim; i++) {
            sizes[i]=output_dimensions[i];
            strides[i] = str;
            str *= sizes[i];
        }
        dnnLayout_t layout;
        dnnLayoutCreate_F32(&layout, dim, sizes, strides);
        void *data;
        void *gradient;
        dnnAllocateBuffer_F32(&data, layout);
        dnnAllocateBuffer_F32(&gradient, layout);
        dnnLayoutDelete_F32(layout);
        data_tensors_.push_back(data);
        gradient_tensors_.push_back(gradient);
    }
    // when all the primitives and the neighboring buffers are ready
    // the primitives are given the pointers to the buffers
    for (int i = 0; i < net_.size(); i++) {
        net_[i]->setFwdInput(data_tensors_[i]);
        net_[i]->setFwdOutput(data_tensors_[i+1]);
        net_[i]->setBwdInput(gradient_tensors_[i+1]);
        net_[i]->setBwdOutput(gradient_tensors_[i]);
    }
    // initialize each primitive's weights
    Initializer init;
    for (int i = 0; i < net_.size(); i++) {
        net_[i]->initialize(&init);
    }
}
void SequentialNetwork::train(void *X, vector<size_t> const &truth, Optimizer *o) {
    SoftMaxObjective obj;
    for (int i = 0; i < 1000; i++) {
        forward(X+i*channel_*height_*width_);
        getLoss(&obj, truth);
        backward();
        update(o, 0.001f);
    }
}
void SequentialNetwork::forward(void *X) {
    data_tensors_[0] = X;
    net_[0]->setFwdInput(X);
    int count = 0;
    for (auto &layer : net_) {
        std::cout << "Layer Forward: " << count << std::endl;
        layer->forward();
        count++;
    }
}
float SequentialNetwork::getLoss(SoftMaxObjective *obj, vector<size_t> const &truth) {
    return obj->computeLossAndGradient(batch_size_,
                                       classes_,
                                       (float *) data_tensors_[data_tensors_.size()-1],
                                       truth,
                                       (float *) gradient_tensors_[gradient_tensors_.size()-1]);
}
void SequentialNetwork::backward() {
    for (auto &layer : net_) {
        layer->backward();
    }
}
void SequentialNetwork::update(Optimizer *opt, float learning_rate) {
    for (auto &layer : net_) {
        layer->update(opt, learning_rate);
    }
}
