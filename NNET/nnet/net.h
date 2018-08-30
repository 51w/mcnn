#pragma once
#include "nnet/layer.h"
#include "nnet/blob.h"

//#define  USE_ModelH
#ifdef   USE_ModelH
#include "yolo3_model.h"
#endif

namespace NNET{
	
class Net
{
public:
	Net( ){}; //init
	~Net();
	
	int ncnn_param(const char* protopath);
	int number_param();
  
	int Yolo_model(FILE* fp);
	int ncnn_model(FILE* fp);
	int load_model(const char* modelpath, int type = 0);
  
	void SetParam(string paramstr, int layer_id);
	
	void Run(); //compute

public:
	int _NH, _NW, _NC;
	int   _Nclass;  //Yolo-v3
	float _Nthresh; //Yolo-v3
	
	vector<Blob*> net_input_blobs_;
	vector<Blob*> net_output_blobs_;	
	vector<vector<Blob*> > bottom_vecs_;
	vector<vector<Blob*> > top_vecs_;
	
//protected:
	vector<shared_ptr<Blob> > blobs_;
	vector<shared_ptr<Layer> > layers_;
	
	vector<vector<string>> _param;
	vector<string> layer_names_;
	vector<string> blob_names_;
};

}