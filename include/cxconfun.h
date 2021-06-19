// cxconfun.h
#pragma once
// Copyright Richard Ansorge 2020. 
// Part of cx utility suite devevoped for CUP Publication:
//         Programming in Parallel with CUDA 

//=======================================================================
// small set of constexpr math functions to be  evaluated at compile time
// Power series for sin and cos good for angles in [-2pi,2pi]
// tan function cuts off singualarity at 10^9. Need to specify a fixed
// number of terms or iterations to keep compiler happy.
//=======================================================================
namespace cx {
	constexpr int fact(int n)  // just for fun
	{
		int k = 1;
		int f = k;
		while(k <= n) f *= k++;
		return f;
	}
	constexpr double sin_cx(double x)
	{
		double s = x;
		int nit = 1;
		double fnit = 1.0;
		double term = x;
		while(nit < 12) {
			term *= -x * x / (2.0*fnit*(2.0*fnit + 1.0));
			s += term;
			nit++;
			fnit++;
		}
		return s;
	}
	constexpr double cos_cx(double x)
	{
		double s = 1;
		int nit = 1;
		double fnit = 1.0;
		double term = 1.0;
		while(nit < 12) {
			term *= -x * x / (2.0*fnit*(2.0*fnit - 1.0));
			s += term;
			nit++;
			fnit++;
		}
		return (float)s;
	}
	constexpr double tan_cx(double x)
	{
		double s = sin_cx(x);
		double c = cos_cx(x);
		double t = 0.0;
		if(c > 1.0e-9 || c < -1.0e-09) t = s / c;
		else if(c >= 0.0) t = s / 1.0e-09;
		else t = -s / 1.0e-09;
		return t;
	}

	constexpr double sqrt_cx(double x)
	{
		// return root of abs
		if(x < 0) x = -x;
		// NB sqrt(x) > x if x < 1
		float step = (x >= 1.0) ? x/2.0 : 0.5;
		float s =    (x >= 1.0) ? x/2.0 : x;
		int nit = 32;
		while(nit >0) {
			if(s*s > x) s -= step;
			else        s += step;
			step *= 0.5;
			nit--;
		}
		return s;
	}
}
//  end file cxconfun.h
