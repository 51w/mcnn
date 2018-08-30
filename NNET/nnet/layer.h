#pragma once
#include "nnet/blob.h"
#include "nnet/layer_factory.h"
#include "nnet/net.h"

#include <string>
#include <vector>

namespace NNET
{

class Net;
class Layer 
{
public:
	// 初始化layer_param_
	explicit Layer(const vector<string>& param) : layer_param_(param) {}		 
	//virtual ~Layer() {LOG(INFO) << "layer";}
	
	// 返回Layer类型
	virtual inline const char* type() const { return "";}
	
	void xSetUp(Tensor& Input, Tensor& Output, Net *net){
		SetUp(Input, Output);
		L2Net(net);
		Reshape(Input, Output);
	}
	virtual void L2Net(Net *net){}
	
	virtual void SetUp(Tensor& Input, Tensor& Output){}
	
	virtual void Run(Tensor& Input, Tensor& Output) = 0;
	
	virtual void Reshape(Tensor& Input, Tensor& Output) = 0;
	
	
	int GetParam_Int32(int id, int val)
	{
		return layer_param_.size()>id ? stoi(layer_param_[id]) : val;
	}	
	float GetParam_f32(int id, float val)
	{
		return layer_param_.size()>id ? stof(layer_param_[id]) : val;
	}
	int GetParam_vec(int id, vector<float> &val)
	{
		CHECK(layer_param_.size() > id);
		string ss = layer_param_[id];
		
		int npos = 0;
		int index = ss.find(',', npos);
		if(index != string::npos)
		{
			string name = ss.substr(0, index);
			int num = stoi(name);
			for(int i=0; i<num-1; i++)
			{
				npos = index+1;
				index = ss.find(',', npos);
				CHECK(index != string::npos);
				name = ss.substr(npos, index);
				val.push_back(stof(name));
			}
			name = ss.substr(index+1, ss.size());
			val.push_back(stof(name));
		}
		else
		{
			int num = stoi(ss);
			CHECK(num == 0);
		}
	}
	
public:
	vector<string> layer_param_; // 输入参数
	vector<shared_ptr<Blob>> blobs_; // 权重
};

}