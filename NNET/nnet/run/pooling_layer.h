#pragma once
#include "nnet/layer.h"

namespace NNET
{

class PoolingLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Pooling);

protected:
	int XC, YC;
	int XH, YH;
	int XW, YW;
	
	int pooling_type;
    int kernel;
    int stride;
    int pad;
	
    int global_pooling;
    int pad_mode;// 0=full 1=valid 2=SAME
	vector<int> mask;
	
	enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };
};

REGISTER_LAYER_CLASS(Pooling);
}