#pragma once
#include <string>
#include <vector>

#include "cnn/blob.hpp"
#include "cnn/layer_factory.hpp"

namespace H2O {

class Layer 
{
public:
	explicit Layer(const vector<string>& param) 
		: layer_param_(param) {}		
	
	virtual ~Layer() {}

	virtual inline const char* type() const 
	{  return "";  }
	
	void SetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
	{  LayerSetUp(bottom, top);  }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {}
	
	inline  void Forward(const vector<Blob*>& bottom, const vector<Blob*>& top);
	
	virtual void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;

	
	virtual int Get_YoloParam() {}

//protected:
	vector<string> layer_param_;
	vector<shared_ptr<Blob> > blobs_;
	int NetH = 0;
	int NetW = 0;
	float _NetThresh = 0;
	
};

inline void Layer::Forward(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	Forward_cpu(bottom, top);
}

}