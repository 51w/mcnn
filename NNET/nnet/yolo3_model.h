#ifdef  USE_ModelH

#pragma once
#include <vector>
#include <string>
namespace NNET{

inline int GetNumber_Net(std::vector<std::vector<int> >& info)
{
	std::vector<int> layer;
	
	layer.push_back(0); //type
	layer.push_back(0); //name
	layer.push_back(0); //bott
	layer.push_back(1); //top
	layer.push_back(0);
	info.push_back(layer);
	layer.clear();
	
	
	layer.push_back(1); //type
	layer.push_back(1); //name
	layer.push_back(1); //bott
	layer.push_back(1); //top
	layer.push_back(0);
	layer.push_back(1);
	info.push_back(layer);
	layer.clear();
	

	
	return 0;
}

}
#endif