#include "stopwatch.h"
StopWatch::StopWatch() : state_(Stopped),  
                         start_time_(0.0), 
                         stop_time_(0.0) { }
void StopWatch::pressPrimaryButton() {
    switch (state_) {
        case Stopped: {
            start_time_ = omp_get_wtime();
            state_ = Running;
            splits_.push_back(omp_get_wtime());
            break;
        }
        case Running: {
            stop_time_ = omp_get_wtime();
            state_ = Stopped;
            break;
        }
        default: break;
    }
};
void StopWatch::pressSecondaryButton() {
    switch (state_) {
        case Stopped: {
            start_time_ = 0.0;
            stop_time_ = 0.0;
            splits_.erase(splits_.begin());
            break;
        }
        case Running: {
            splits_.push_back(omp_get_wtime());
            break;
        }
        default: break;
    }   
}
double StopWatch::getTime() const {
    switch (state_) {
        case Stopped: {
            return stop_time_ - start_time_;
            break;
        }
        case Running: {
            return omp_get_wtime() - start_time_;
            break;
        }
        default: break;
    }
}
std::vector<double> StopWatch::getLapTimes() const {
    std::vector<double> lap_times(splits_.size()-1);
    for (int i = 0; i < lap_times.size(); i++) {
        lap_times[i] = splits_[i+1] - splits_[i];
    }
    return lap_times;
}
