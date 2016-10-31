#ifndef STOPWATCH_H
#define STOPWATCH_H
#include "omp.h"
#include <vector>
class StopWatch {
  public:
    StopWatch();
    // press to start and stop
    void pressPrimaryButton();
    // while the watch is running, press to save lap time
    // while not running, press to reset
    void pressSecondaryButton();
    // if it's running, timer up to now
    // if stopped, timer for when it was stopped
    double getTime() const;
    // get the lap times up to now
    std::vector<double> getLapTimes() const;
  private:
    enum State {
        Stopped,
        Running
    };
    State state_;
    double start_time_;
    double stop_time_;
    std::vector<double> splits_;
};
#endif // STOPWATCH_H
