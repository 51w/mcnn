#pragma once
#include "nnet/layer.h"

namespace NNET
{

class ReshapeLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(ReshapeLayer);

protected:
	int dim[3];
	int permute;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(ReshapeLayer);
}