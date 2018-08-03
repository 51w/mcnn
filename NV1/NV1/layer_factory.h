#pragma once
#include <functional>
#include "layer.h"

namespace NV1{

class Layer;
class LayerRegistry 
{
public:
	typedef shared_ptr<Layer> (*Creator)(const vector<string>&);
	typedef map<string, Creator> CreatorRegistry;

	static CreatorRegistry& Registry() 
	{
		static CreatorRegistry* g_registry_ = new CreatorRegistry();
		return *g_registry_;
	}
	
	
	//*********************//
	static void AddCreator(const string& type, Creator creator) 
	{
		CreatorRegistry& registry = Registry();
		if(registry.count(type) != 0)
		{
			fprintf(stderr, "Layer type %s already registered.\n", type.c_str());
			exit(0);
		}
		registry[type] = creator;
	}
	static shared_ptr<Layer> CreateLayer(const string& name, const string& type, const vector<string>& param) 
	{	
		CreatorRegistry& registry = Registry();
		if(registry.count(type) != 1)
		{
			string typelist = LayerTypeListString();
			fprintf(stderr, "\nUnknown layer type: %s\n(known types: %s )\n", type.c_str(), typelist.c_str());
			exit(0);
		}
		return registry[type](param);
	}
	//*********************//
	
	
	// 注册的Layer
	static vector<string> LayerTypeList() {
		CreatorRegistry& registry = Registry();
		vector<string> layer_types;
		for (typename CreatorRegistry::iterator iter = registry.begin();
			iter != registry.end(); ++iter) {
			layer_types.push_back(iter->first);
		}
		return layer_types;
	}

private:
	LayerRegistry() {}
	
	// 所有的Layer <--> 输出到一个string
	static string LayerTypeListString() {
		vector<string> layer_types = LayerTypeList();
		string layer_types_str;
		for (vector<string>::iterator iter = layer_types.begin();
			iter != layer_types.end(); ++iter) {
			if (iter != layer_types.begin()) {
				layer_types_str += ", ";
			}
			layer_types_str += *iter;
		}
		return layer_types_str;
	}
};


class LayerRegisterer 
{
public:
	LayerRegisterer(const string& type, LayerRegistry::Creator creator)
	{
		LayerRegistry::AddCreator(type, creator);
	}
};


// 注册Layer
#define REGISTER_LAYER_CREATOR(type, creator)                 \
	static LayerRegisterer g_creator_##type(#type, creator);  \

 
#define REGISTER_LAYER_CLASS(type)                                		 \
	shared_ptr<Layer> Creator_##type##Layer(const vector<string>& param) \
	{                                                                  	 \
		return shared_ptr<Layer>(new type##Layer(param));           	 \
	}                                                                  	 \
	REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)					 \


#define Tensor const vector<Blob*> 

// SetUp <--> Run  
#define REGISTER_LAYER_INIT(name) 						\
  	explicit name##Layer(const vector<string>& param) 	\
		: Layer(param) {} 								\
	virtual inline const char* type() const 			\
	{ return #name; } 									\
	virtual void SetUp(Tensor& bottom, Tensor& top); 	\
	virtual void  Run(Tensor& bottom, Tensor& top); 	\
 
}