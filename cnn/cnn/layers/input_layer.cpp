#include "cnn/layers/input_layer.hpp"

namespace cnn 
{

void InputLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int w = stoi(this->layer_param_[0]);
	int h = stoi(this->layer_param_[1]);
	int c = stoi(this->layer_param_[2]);
	
	for (int i = 0; i < top.size(); ++i) 
	{
		top[i]->Reshape(c, h, w);
	}
	//fprintf(stderr, "InputLayer: %d %d %d\n", top[0]->channels(), top[0]->height(), top[0]->width());
}

REGISTER_LAYER_CLASS(Input);

}
