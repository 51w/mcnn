#pragma once
#include "cnn/layer.hpp"

namespace H2O {

class RouteLayer : public Layer
{

public:
	explicit RouteLayer(const vector<string>& param)
		: Layer(param) {}
		
	virtual inline const char* type() const 
	{ return "Example"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);	
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
	
};
}