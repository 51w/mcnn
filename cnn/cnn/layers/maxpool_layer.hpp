#pragma once
#include "cnn/layer.hpp"

namespace cnn {

class MaxpoolLayer : public Layer
{
	
public:
	explicit MaxpoolLayer(const vector<string>& param) 
		: Layer(param) {}
		
	virtual inline const char* type() const 
	{ return "Maxpool"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

protected:
	int _kernel;
    int _stride;
    int _pad;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};
}