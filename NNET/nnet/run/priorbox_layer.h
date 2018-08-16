#pragma once
#include "nnet/layer.h"

namespace NNET
{

class PriorBoxLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(PriorBox);

protected:
	vector<float> min_sizes;
	vector<float> max_sizes;
	vector<float> aspect_ratios;
	float variances[4];

	int flip;
    int clip;
    int image_width;
    int image_height;
    float step_width;
    float step_height;
    float offset;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(PriorBox);
}