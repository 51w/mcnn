#pragma once
#include "cnn/layer.hpp"

namespace cnn
{

class Yolo2Layer : public Layer
{
	
public:
	explicit Yolo2Layer(const vector<string>& param)
		: Layer(param) {}
	
	virtual inline const char* type() const 
	{ return "Yolo2"; }
	
	virtual void LayerSetUp( const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

protected:
	float _thresh;
	int _num;
	int _classes;
	int _coords;
	float _biases[10];
};

}