#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Flatten : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Flatten);

protected:

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Flatten);
}