#include "sequential_network.h"
void SequentialNetwork::add_layer(Layer *l) {
    net_.push_back(l);
}
void SequentialNetwork::train() {
    for (int i = 0; i < 1000; i++) {
        forward();
        backward();
        update();
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
void SequentialNetwork::update() {
    for (auto &layer : net_) {
        layer->update();
    }
}
