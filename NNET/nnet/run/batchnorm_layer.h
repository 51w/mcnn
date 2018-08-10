#pragma once
#include "nnet/layer.h"

namespace NNET
{

class BatchNormLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(BatchNorm);

protected:
	int channels;
    float eps;
	
	bool init;
	vector<float> a_data;
    vector<float> b_data;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(BatchNorm);
}