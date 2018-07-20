#pragma once
#include "cnn/layer.hpp"

namespace H2O {

class UpsampleLayer : public Layer
{
public:
	explicit UpsampleLayer(const vector<string>& param) 
		: Layer(param) {}
		
	virtual inline const char* type() const 
	{ return "Example"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
	
protected:	
	int XC, YC;
	int XH, YH;
	int XW, YW;
	
	int _stride;
};
}