#pragma once
#include "nnet/layer.h"

namespace NNET
{

class ReshapeLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Reshape);

protected:
	int permute;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Reshape);
}