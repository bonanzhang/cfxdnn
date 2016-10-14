#include "sequential_network.h"
SequentialNetwork::SequentialNetwork(size_t batch_size, size_t channel, size_t height, size_t width) {
    batch_size_ = batch_size;
    channel_ = channel;
    height_ = height;
    width_ = width;
}
SequentialNetwork::~SequentialNetwork() {
    for (auto r : data_tensors_) {
        dnnReleaseBuffer_F32(r);
    }
    for (auto r : gradient_tensors_) {
        dnnReleaseBuffer_F32(r);
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
    std::vector<size_t> input_dimensions;
    input_dimensions.push_back(width_);
    input_dimensions.push_back(height_);
    input_dimensions.push_back(channel_);
    input_dimensions.push_back(batch_size_);
    for (int i = 0; i < layers_.size(); i++) {
        std::vector<size_t> output_dimensions;
        // each time a primitive is contructed,
        // it requires the input tensor dimensions
        // it gives back the output tensor dimensions,
        // which is used by the next layer as the input
        net_.push_back(new Primitive(layers_[i], input_dimensions, output_dimensions));
        input_dimensions = output_dimensions;
        // the tensor resources are allocated here
        // the data and the gradient tensors have the same dimensions
        // this is WHCN
        size_t sizes[4] = {output_dimensions[0], 
                           output_dimensions[1],
                           output_dimensions[2],
                           output_dimensions[3]};
        size_t strides[4] = {1, sizes[0], sizes[0]*sizes[1], sizes[0]*sizes[1]*sizes[2]};
        dnnLayout_t layout;
        dnnLayoutCreate_F32(&layout, 4, sizes, strides);
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
        net_[i]->setBwdInput(gradient_tensors_[i]);
        net_[i]->setBwdOutput(gradient_tensors_[i+1]);
    }
    // TODO: initialize each primitive's weights
}
void SequentialNetwork::train(void *X, void *y, Optimizer *o) {
    data_tensors_[0] = X;
    for (int i = 0; i < 1000; i++) {
        forward();
        backward();
        update(o, 0.001f);
    }
}
void SequentialNetwork::forward() {
    for (auto &layer : net_) {
        layer->forward();
    }
}
void SequentialNetwork::backward() {
    for (auto &layer : net_) {
        layer->backward();
    }
}
void SequentialNetwork::update(Optimizer *opt, float learning_rate) {
    for (auto &layer : net_) {
        layer->update(opt, 0.001f);
    }
}
