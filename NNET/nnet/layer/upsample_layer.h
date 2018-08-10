#pragma once
#include "nnet/layer.h"

namespace NNET
{

class UpsampleLayer : public Layer
{

public:
	REGISTER_LAYER_INIT(Upsample);
	
protected:
	int _stride;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Upsample);
}