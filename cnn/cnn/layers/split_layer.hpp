#pragma once
#include "cnn/layer.hpp"

namespace cnn 
{

class SplitLayer : public Layer
{
	
public:
	explicit SplitLayer(const vector<string>& param) 
		: Layer(param) {}
	
	virtual inline const char* type() const 
	{ return "Split"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);
		
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
	
};

}