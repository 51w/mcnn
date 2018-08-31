#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Input : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Input);
	
	virtual void L2Net(Net *net);

protected:
	int YC;
	int YH;
	int YW;
};

REGISTER_LAYER_CLASS(Input);
}