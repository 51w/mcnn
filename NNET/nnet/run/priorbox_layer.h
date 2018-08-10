#pragma once
#include "nnet/layer.h"

namespace NNET
{

class PriorBoxLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(PriorBox);

protected:
	

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(PriorBox);
}