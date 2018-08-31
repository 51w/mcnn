#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Permute : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Permute);

protected:
	int order_type;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Permute);
}