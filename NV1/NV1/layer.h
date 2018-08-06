#pragma once
#include "NV1/blob.h"
#include "NV1/layer_factory.h"
#include "NV1/net.h"

#include <string>
#include <vector>

namespace NV1{

class Net;
class Layer 
{
public:
	// 初始化layer_param_
	explicit Layer(const vector<string>& param) : layer_param_(param) {}		 
	virtual ~Layer() {}
	
	// 返回Layer类型
	virtual inline const char* type() const { return "";}
	
	
	void xSetUp(Tensor& Input, Tensor& Output, Net *net){
		SetUp(Input, Output);
		L2Net(net);
	}
	virtual void L2Net(Net *net){}
	
	virtual void SetUp(Tensor& Input, Tensor& Output){}
	
	virtual void Run(Tensor& Input, Tensor& Output) = 0;
	
public:
	vector<string> layer_param_; // 输入参数
	vector<shared_ptr<Blob>> blobs_; // 权重
};

}