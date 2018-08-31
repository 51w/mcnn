#pragma once
#include "nnet/layer.h"

namespace NNET
{

class ReLU : public Layer
{
	
public:
	REGISTER_LAYER_INIT(ReLU);

protected:
	float slope;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(ReLU);
}