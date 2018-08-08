#pragma once
#include "NV1/layer.h"

namespace NV1{

class SoftmaxLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Softmax);

protected:
	int axis;
	int outer_num_;
	int inner_num_;
	
	int YC;
	int YH;
	int YW;
	
	Blob sum_multiplier_;
	Blob scale_;
};

REGISTER_LAYER_CLASS(Softmax);
}