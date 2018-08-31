#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Eltwise : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Eltwise);

protected:
	int op_type;
	int coeff_size;
	vector<float> coeffs;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
	enum{ Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2 };
};

REGISTER_LAYER_CLASS(Eltwise);
}