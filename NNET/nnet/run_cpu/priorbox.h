#pragma once
#include "nnet/layer.h"

namespace NNET
{

class PriorBox : public Layer
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

	int num_prior;
	int X1C, X0C;
	int X1H, X0H;
	int X1W, X0W;
};

REGISTER_LAYER_CLASS(PriorBox);
}