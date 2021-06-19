// file cxtimers.h
#pragma once
// Copyright Richard Ansorge 2020. 
// Part of cx utility suite devevoped for CUP Publication:
//         Programming in Parallel with CUDA 

// provdides a MYTimer object for host bast elapsed time measurements.
// The timer depends on the C++ <chrono>
// usage: 
// (A) lap_ms() to returns interval since previous lap_ms(), start or reset.
// (B) use start() and add() pairs to accumaulate one or more time spans.
//     Then time() for accumulated duration in ms.

#include <cstdio>
#include <cstdlib>
#include <chrono>

namespace cx {
	class timer {
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> tstart;
		std::chrono::time_point<std::chrono::high_resolution_clock> lap;
		double duration;  // accumulates time spans between start/add pairs.
	public:
		timer() {
			tstart = std::chrono::high_resolution_clock::now();
			lap = tstart;
			duration = 0.0;
		}
	
		void start() { 
			lap = std::chrono::high_resolution_clock::now();
			return; 
		}
		
		void add() {
			auto old_lap = lap;
			lap = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::milli> time_span = (lap - old_lap);
			duration += (double)time_span.count();
		}
		void reset()  { 
			lap = std::chrono::high_resolution_clock::now();
			duration = 0.0;  
		}
		double time() { return duration; }

		double lap_ms()
		{
			auto old_lap = lap;
			lap = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::milli> time_span = (lap - old_lap);
			return (double)time_span.count();
		}

		double lap_ns()
		{
			auto old_lap = lap;
			lap = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::nano> time_span = (lap - old_lap);
			return (double)time_span.count();
		}
		double lap_us()
		{
			auto old_lap = lap;
			lap = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::micro> time_span = (lap - old_lap);
			return (double)time_span.count();
		}
		double spot_ms()
		{
			auto spot = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::milli> time_span = (spot - tstart);
			return (double)time_span.count();
		}
		double spot_us()
		{
			auto spot = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::micro> time_span = (spot - tstart);
			return (double)time_span.count();
		}
		double spot_ns()
		{
			auto spot = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double,std::nano> time_span = (spot - tstart);
			return (double)time_span.count();
		}

	};
}