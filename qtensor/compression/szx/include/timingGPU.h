#ifndef __TIMING_CUH__
#define __TIMING_CUH__

/**************/
/* TIMING GPU */
/**************/

// Events are a part of CUDA API and provide a system independent way to measure execution times on CUDA devices with approximately 0.5
// microsecond precision.

struct PrivateTimingGPU;

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:

        TimingGPU();

        ~TimingGPU();

        void StartCounter();
        void StartCounterFlags();

        float GetCounter();

}; // TimingGPU class

#endif
