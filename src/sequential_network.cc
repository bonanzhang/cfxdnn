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
    for (int i = 1; i < data_tensors_.size(); i++) {
        if (data_tensors_[i] != nullptr) {
            dnnReleaseBuffer_F32(data_tensors_[i]);
        }
    }
    for (int i = 1; i < gradient_tensors_.size(); i++) {
        if (gradient_tensors_[i] != nullptr) {
            dnnReleaseBuffer_F32(gradient_tensors_[i]);
        }
    }
    for (auto p : net_) {
        delete p;
    }
    for (auto l : layers_) {
        delete l;
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
        // the current convolution primitive used needs a padding layer
        // but this is going to be a transparent layer from the users
        // the users already did all the work they need to do
        // by specifying the padding sizes
        std::vector<size_t> padding_size;
        if (layers_[i]->needsPadding(padding_size)) {
            // TODO: that false means not unpadding for back prop
            // make sure that's always true
            std::vector<size_t> padded_dimensions;
            net_.push_back(new Padder(input_dimensions, padding_size, padded_dimensions, false));
            void *pad_data;
            void *pad_gradient;
            allocateBuffer(padded_dimensions, pad_data);
            allocateBuffer(padded_dimensions, pad_gradient);
            // store the pointers to the buffers
            data_tensors_.push_back(pad_data);
            gradient_tensors_.push_back(pad_gradient);
            //construct the acutal primitive, which requires
            //the original unpadded source dimensions
            for(int d = 0; d < padding_size.size(); d++) {
                padded_dimensions[d] -= 2 * padding_size[d];
            }
            net_.push_back(new Primitive(layers_[i], padded_dimensions, output_dimensions));
        } else {
            net_.push_back(new Primitive(layers_[i], input_dimensions, output_dimensions));
        }
        void *data = nullptr;
        void *gradient = nullptr;
        // the tensor resources are allocated here
        // the data and the gradient tensors have the same dimensions
        // this is WHCN
        allocateBuffer(output_dimensions, data);
        allocateBuffer(output_dimensions, gradient);
        data_tensors_.push_back(data);
        gradient_tensors_.push_back(gradient);
        //next layer's input is this layer's output
        input_dimensions = output_dimensions;
    }
    
    std::cout << "layer " << layers_.size() << std::endl
              << "data  " << data_tensors_.size() << std::endl
              << "grad  " << gradient_tensors_.size() << std::endl
              << "net   " << net_.size() << std::endl;
    // when all the primitives and the neighboring buffers are ready
    // the primitives are given the pointers to the buffers
    for (int i = 0; i < net_.size(); i++) {
        std::cout << "i=" << i << " (forward)  from: " << data_tensors_[i] 
                  << " to: " << data_tensors_[i+1] << std::endl;
        std::cout << "i=" << i << " (backward) from: " << gradient_tensors_[i+1] 
                  << " to: " << gradient_tensors_[i] << std::endl;
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
    std::cout << "input set for i=0 " << X << std::endl;
    std::cout << "forward pass for all " << net_.size() << " net components" << std::endl;
    for (int i = 0; i < net_.size(); i++) {
        std::cout << "net components: " << i << std::endl;
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
        std::cout << "net components: " << i << std::endl;
        net_[i]->backward();
        std::cout << "finished" << std::endl;
    }
}
void SequentialNetwork::update(Optimizer *opt, float learning_rate) {
    std::cout << "update for all " << net_.size() << " net components" << std::endl;
    for (int i = 0; i < net_.size(); i++) {
        std::cout << "net components: " << i << std::endl;
        net_[i]->update(opt, learning_rate);
        std::cout << "finished" << std::endl;
    }
}
void SequentialNetwork::allocateBuffer(vector<size_t> const &dimensions, void * &data) {
    // calculate the buffer sizes
    size_t const dim = dimensions.size();
    size_t sizes[dim];
    size_t str = 1;
    size_t strides[dim];
    for (int d = 0; d < dim; d++) {
        sizes[d] = dimensions[d];
        strides[d] = str;
        str *= sizes[d];
    }
    // allocate the buffers
    dnnError_t e;
    dnnLayout_t layout;
    e = dnnLayoutCreate_F32(&layout, dim, sizes, strides);
    if (e != E_SUCCESS) std::cout << "layout create failed\n";
    e = dnnAllocateBuffer_F32(&data, layout);
    if (e != E_SUCCESS) std::cout << "layout allocate buffer failed\n";
    e = dnnLayoutDelete_F32(layout);
    if (e != E_SUCCESS) std::cout << "layout delete failed\n";
    std::cout << "Allocate A Buffer with dimensions: ";
    for (auto const &i : dimensions) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "At: " << data << std::endl;
}
