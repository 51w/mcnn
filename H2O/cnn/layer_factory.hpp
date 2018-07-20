#pragma once
#include <functional>
#include "layer.hpp"

namespace H2O {

class Layer;

class LayerRegistry {
public:
	typedef shared_ptr<Layer>(*Creator)(const vector<string>&);
	typedef map<string, Creator> CreatorRegistry;

	static CreatorRegistry& Registry() {
		static CreatorRegistry* g_registry_ = new CreatorRegistry();
		return *g_registry_;
	}

	// Adds a creator.
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

	// Get a layer using a LayerParameter.
	static shared_ptr<Layer> CreateLayer(const string& name, 
		const string& type, const vector<string>& param) 
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


class LayerRegisterer {
public:
	LayerRegisterer(const string& type,
		shared_ptr<Layer>(*creator)(const vector<string>&)) 
	{
		// LOG(INFO) << "Registering layer type: " << type;
		LayerRegistry::AddCreator(type, creator);
	}
};


#define REGISTER_LAYER_CREATOR(type, creator)               \
  static LayerRegisterer g_creator_##type(#type, creator);	

 
#define REGISTER_LAYER_CLASS(type)                                		\
  shared_ptr<Layer> Creator_##type##Layer(const vector<string>& param) 	\
  {                                                                  	\
    return shared_ptr<Layer>(new type##Layer(param));           		\
  }                                                                  	\
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}