#pragma once
#include "cnn/layer.hpp"

namespace cnn 
{

class ConvolutionLayer : public Layer
{
	
public:
	explicit ConvolutionLayer(const vector<string>& param) 
		: Layer(param){}

	virtual inline const char* type() const 
	{ return "Convolution"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	void bias_inplace(Blob& bottom);
	void scale_inplace(Blob& bottom);
	void normalize_inplace(Blob& bottom);
	void activate_inplace(Blob& bottom, int type);

protected:	
	int  _kernel;
	int  _pad;
	int  _stride;
	int  _filters;
	int  _activation;
	bool _normalize;
	
	Blob _im2col;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;	
	Blob blob_bordered;
};

}