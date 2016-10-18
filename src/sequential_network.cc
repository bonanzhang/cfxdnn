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
    int component_index = 0;
    for (int i = 0; i < layers_.size(); i++) {
        vector<size_t> output_dimensions;
        // each time a primitive is contructed,
        // it requires the input tensor dimensions
        // it gives back the output tensor dimensions,
        // which is used by the next layer as the input
        // the current convolution primitive used needs a padding layer
        // but this is going to be a transparent layer from the users
        // the users already did all the work they need to do
        // by specifying the padding sizes
        std::vector<size_t> padding_size;
        std::vector<size_t> padded_dimensions;
        if (layers_[i]->needsPadding(padding_size)) {
            // TODO: that false means not unpadding for back prop
            // make sure that's always true
            // construct the padder and allocate its buffers
            net_.push_back(new Padder(input_dimensions, padding_size, padded_dimensions, false));
            std::cout << "added component " << component_index++ << std::endl;
            for (auto const d : input_dimensions) std::cout << d << " ";
            std::cout << " >> ";
            for (auto const d : padded_dimensions) std::cout << d << " ";
            std::cout << std::endl;
            // calculate the buffer sizes
            size_t const pad_dim = padded_dimensions.size();
            size_t pad_sizes[pad_dim];
            size_t pad_str = 1;
            size_t pad_strides[pad_dim];
            for (int d = 0; d < pad_dim; d++) {
                pad_sizes[d] = padded_dimensions[d];
                pad_strides[d] = pad_str;
                pad_str *= pad_sizes[d];
            }
            // allocate the buffers
            dnnLayout_t pad_layout;
            dnnLayoutCreate_F32(&pad_layout, pad_dim, pad_sizes, pad_strides);
            void *pad_data;
            void *pad_gradient;
            dnnAllocateBuffer_F32(&pad_data, pad_layout);
            dnnAllocateBuffer_F32(&pad_gradient, pad_layout);
            dnnLayoutDelete_F32(pad_layout);
            // store the pointers to the buffers
            data_tensors_.push_back(pad_data);
            gradient_tensors_.push_back(pad_gradient);
            //construct the acutal primitive
            net_.push_back(new Primitive(layers_[i], padded_dimensions, output_dimensions));
            std::cout << "added component " << component_index++ << std::endl;
            for (auto const d : padded_dimensions) std::cout << d << " ";
            std::cout << " >> ";
            for (auto const d : output_dimensions) std::cout << d << " ";
            std::cout << std::endl;
        } else {
            net_.push_back(new Primitive(layers_[i], input_dimensions, output_dimensions));
            std::cout << "added component " << component_index++ << std::endl;
            for (auto const d : input_dimensions) std::cout << d << " ";
            std::cout << " >> ";
            for (auto const d : output_dimensions) std::cout << d << " ";
            std::cout << std::endl;
        }
        input_dimensions = output_dimensions;
        // the tensor resources are allocated here
        // the data and the gradient tensors have the same dimensions
        // this is WHCN
        size_t const dim = output_dimensions.size();
        size_t sizes[dim];
        size_t str = 1;
        size_t strides[dim];
        for (int d = 0; d < dim; d++) {
            sizes[d]=output_dimensions[d];
            strides[d] = str;
            str *= sizes[d];
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
        forward(((float *) X) + i*channel_*height_*width_);
        getLoss(&obj, truth);
        backward();
        update(o, 0.001f);
    }
}
void SequentialNetwork::forward(void *X) {
    data_tensors_[0] = X;
    net_[0]->setFwdInput(X);
    std::cout << "forward pass for all " << net_.size() << " net components" << std::endl;
    for (int i = 0; i < net_.size(); i++) {
        std::cout << "net components: " << i << "...";
        net_[i]->forward();
        std::cout << "finished" << std::endl;
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
    std::cout << "backard pass for all " << net_.size() << " net components" << std::endl;
    for (int i = net_.size()-1; i >= 0; i--) {
        std::cout << "net components: " << i << "...";
        net_[i]->backward();
        std::cout << "finished" << std::endl;
    }
}
void SequentialNetwork::update(Optimizer *opt, float learning_rate) {
    std::cout << "update for all " << net_.size() << " net components" << std::endl;
    for (int i = 0; i < net_.size(); i++) {
        std::cout << "net components: " << i << "...";
        net_[i]->update(opt, learning_rate);
        std::cout << "finished" << std::endl;
    }
}
