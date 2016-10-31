#include "sequential_network.h"
SequentialNetwork::SequentialNetwork(size_t batch_size, size_t channel, 
                                     size_t height, size_t width, 
                                     size_t classes)
    : batch_size_(batch_size), channel_(channel),
      height_(height), width_(width), classes_(classes) {}
SequentialNetwork::~SequentialNetwork() {
    for (int i = 1; i < data_tensors_.size(); i++) {
        if (data_tensors_[i] != nullptr) {
            dnnReleaseBuffer_F32(data_tensors_[i]);
//            std::cout << "releasing data buffer at: " 
//                      << static_cast<void*>(data_tensors_[i]) << std::endl;
        }
    }
    for (int i = 0; i < gradient_tensors_.size(); i++) {
        if (gradient_tensors_[i] != nullptr) {
            dnnReleaseBuffer_F32(gradient_tensors_[i]);
//            std::cout << "releasing gradient buffer at: " 
//                      << static_cast<void*>(gradient_tensors_[i]) << std::endl;
        }
    }
    for (auto p : net_) {
//        std::cout << "deleting component at: " << static_cast<void*>(p) << std::endl;
        delete p;
    }
    for (auto l : layers_) {
//        std::cout << "deleting layer at: " << static_cast<void*>(l) << std::endl;
        delete l;
    }
}
int SequentialNetwork::add_layer(Layer *l) {
    int id = layers_.size();
    layers_.push_back(l);
//    std::cout << "addr of layer: " << static_cast<void*>(l) << std::endl;
    return id;
}
void SequentialNetwork::finalize_layers() {
    // the first resource is temporarily empty
    // it will be set by train
    void *data = nullptr;
    data_tensors_.push_back(data);
    vector<size_t> input_dimensions{width_, height_, channel_, batch_size_};
    void *gradient = nullptr;
    allocateBuffer(input_dimensions, gradient);
    gradient_tensors_.push_back(gradient);
    for (int i = 0; i < layers_.size(); i++) {
        // each time a primitive is contructed,
        // it requires the input tensor dimensions
        // it gives back the output tensor dimensions,
        // which is used by the next layer as the input
        Primitive * p = new Primitive(layers_[i], input_dimensions);
        vector<size_t> output_dimensions = p->getOutputDimensions();
        net_.push_back(p);
        // the tensor resources are allocated here
        // the data and the gradient tensors have the same dimensions
        // this is WHCN
        allocateBuffer(output_dimensions, data);
        data_tensors_.push_back(data);
        allocateBuffer(output_dimensions, gradient);
        gradient_tensors_.push_back(gradient);
        //next layer's input is this layer's output
        input_dimensions = output_dimensions;
    }
//    for (auto p : net_) {
//        std::cout << p->getComponentName() 
//                  << " allocated at: " << static_cast<void*>(p) << std::endl;
//    }
//    std::cout << "layer " << layers_.size() << std::endl
//              << "data  " << data_tensors_.size() << std::endl
//              << "grad  " << gradient_tensors_.size() << std::endl
//              << "net   " << net_.size() << std::endl;
    // when all the primitives and the neighboring buffers are ready
    // the primitives are given the pointers to the buffers
    for (int i = 0; i < net_.size(); i++) {
//        std::cout << "i=" << i << " (forward)  from: " << data_tensors_[i] 
//                  << " to: " << data_tensors_[i+1] << std::endl;
//        std::cout << "i=" << i << " (backward) from: " << gradient_tensors_[i+1] 
//                  << " to: " << gradient_tensors_[i] << std::endl;
        net_[i]->setFwdInput(data_tensors_[i]);
        net_[i]->setFwdOutput(data_tensors_[i+1]);
        net_[i]->setBwdInput(gradient_tensors_[i+1]);
        net_[i]->setBwdOutput(gradient_tensors_[i]);
    }
    // initialize each primitive's weights
    std::cout << "Initializing weights..." << std::flush;
    Initializer init;
    for (int i = 0; i < net_.size(); i++) {
        net_[i]->initialize(init);
    }
    std::cout << "done" << std::endl;
}
void SequentialNetwork::train(void *X, vector<size_t> const &truth, Optimizer const &o) {
    SoftMaxObjective obj;
    for (int i = 0; i < 1000; i++) {
        forward(static_cast<float *>(X) + i*channel_*height_*width_);
        getLoss(obj, truth);
        backward();
        update(o, 0.001f);
    }
}
void SequentialNetwork::forward(void *X) {
    data_tensors_[0] = X;
    net_[0]->setFwdInput(X);
//    std::cout << "input set for i=0 " << X << std::endl;
//    std::cout << "forward pass for all " << net_.size() << " net components" << std::endl;
    for (int i = 0; i < net_.size(); i++) {
//        std::cout << "net components: " << i << " (" << net_[i]->getComponentName() << ")" <<std::endl;
        net_[i]->forward();
    }
}
float SequentialNetwork::getLoss(SoftMaxObjective const &obj, vector<size_t> const &truth) {
    // TODO: size of truth is expected to be the same as the batch size
    return obj.computeLossAndGradient(
      batch_size_, classes_,
      static_cast<float *>(data_tensors_[data_tensors_.size()-1]),
      truth,
      static_cast<float *>(gradient_tensors_[gradient_tensors_.size()-1]));
}
void SequentialNetwork::backward() {
//    std::cout << "backard pass for all " << net_.size() << " net components" << std::endl;
    for (int i = net_.size()-1; i >= 0; i--) {
        std::cout << "net components: " << i << std::endl;
        net_[i]->backward();
    }
}
void SequentialNetwork::update(Optimizer const &opt, float learning_rate) {
//    std::cout << "update for all " << net_.size() << " net components" << std::endl;
    for (int i = 0; i < net_.size(); i++) {
        std::cout << "net components: " << i << std::endl;
        net_[i]->update(opt, learning_rate);
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
    size_t n = dnnLayoutGetMemorySize_F32(layout)/sizeof(float);
    for (size_t i = 0; i < n; i++) {
        static_cast<float *>(data)[i] = 0.0f;
    }
    e = dnnLayoutDelete_F32(layout);
    if (e != E_SUCCESS) std::cout << "layout delete failed\n";
//    std::cout << "Allocate a buffer with dimensions: ";
//    for (auto const &i : dimensions) {
//        std::cout << i << " ";
//    }
//    std::cout << "Total buffer size: " 
//              << 4*std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>()) 
//              << " bytes" << std::endl;
//    std::cout << "At: " << data << std::endl;
}
