#pragma once
#include "nnet/layer.h"

namespace NNET
{

class ConcatLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Concat);

protected:
	int count_;
	int num_concats_;
	int concat_input_size_;
	int concat_axis_;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Concat);
}