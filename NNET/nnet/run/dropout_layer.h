#pragma once
#include "nnet/layer.h"

namespace NNET
{

class DropoutLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Dropout);

protected:
	float scale;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Dropout);
}