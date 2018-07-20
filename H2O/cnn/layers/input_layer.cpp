#include "cnn/layers/input_layer.hpp"

namespace H2O {

void InputLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	YW = stoi(this->layer_param_[0]);
	YH = stoi(this->layer_param_[1]);
	YC = stoi(this->layer_param_[2]);

	top[0]->Reshape(YC, YH, YW);
}

REGISTER_LAYER_CLASS(Input);
}