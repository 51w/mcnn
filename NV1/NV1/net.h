#pragma once
#include "NV1/layer.h"
#include "NV1/blob.h"

//#define  USE_ModelH
#ifdef   USE_ModelH
#include "yolo3_model.h"
#endif

namespace NV1{
	
class Net
{
public:
	Net( ){}; //init
	~Net(){};
	
	int ncnn_param(const char* protopath);
	int number_param();
  
	int Yolo_model(FILE* fp);
	int load_model(const char* modelpath);
  
	void SetParam(string paramstr);
	
	void Run(); //compute

public:
	int _NH, _NW, _NC;
	int   _Nclass;  //Yolo-v3
	float _Nthresh; //Yolo-v3
	
	vector<Blob*> net_input_blobs_;
	vector<Blob*> net_output_blobs_;	

protected:	
	vector<shared_ptr<Layer> > layers_;
	vector<shared_ptr<Blob> > blobs_;

	vector<vector<Blob*> > bottom_vecs_;
	vector<vector<Blob*> > top_vecs_;
	vector<vector<string>> _param;
};

}