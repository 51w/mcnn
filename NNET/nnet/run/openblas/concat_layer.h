#pragma once
#include "nnet/layer.h"

namespace NNET
{

class ConcatLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Concat);

protected:
	//int count_;
	int concat_num;
	int concat_size;
	int concat_axis;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Concat);
}