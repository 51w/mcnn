#pragma once
#include "cnn/layer.hpp"

namespace cnn{

class InputLayer : public Layer
{
public:
	explicit InputLayer(const vector<string>& param)
		: Layer(param) {}
	
	virtual inline const char* type() const 
	{ return "Input"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);

	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
	{ }
	
};
}