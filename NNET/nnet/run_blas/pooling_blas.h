#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Pooling : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Pooling);
	
protected:
	int pooling_type;
    int kernel;
    int stride;
    int pad;
    int global_pooling;
    int pad_mode;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
	enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };
};

REGISTER_LAYER_CLASS(Pooling);
}