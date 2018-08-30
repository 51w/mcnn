#pragma once
#include "nnet/layer.h"

namespace NNET
{

class ScaleLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Scale);

protected:
	int scale_size;
    int bias_term;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Scale);
}