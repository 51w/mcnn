#pragma once
#include "cnn/layer.hpp"

namespace H2O{

class Yolo3Layer : public Layer
{

public:
	explicit Yolo3Layer(const vector<string>& param)
		: Layer(param) {}
	
	virtual inline const char* type() const 
	{ return "Yolo3"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

protected:
	float _thresh;
	int _num;
	int _classes;
	
	int _mask[3];
	int _anchors[12];
};
}