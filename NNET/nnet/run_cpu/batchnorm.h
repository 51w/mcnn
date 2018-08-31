#pragma once
#include "nnet/layer.h"

namespace NNET
{

class BatchNorm : public Layer
{
	
public:
	REGISTER_LAYER_INIT(BatchNorm);

protected:
	vector<float> a_data;
    vector<float> b_data;
	int channels;
    float eps;
	bool init;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(BatchNorm);
}