#pragma once
#include "cnn/layer.hpp"
#include "cnn/blob.hpp"

namespace cnn
{

class Net
{
public:
	Net(); //init
	~Net();
  
	int load_param(FILE* fp);
	int load_param(const char* protopath);
  
	int load_model(FILE* fp);
	int load_model(const char* modelpath);
  
	void SetParam(string paramstr);
	
	void forward(); //compute
	
	bool has_blob(const string& blob_name) const;
	const shared_ptr<Blob> blob_by_name(const string& blob_name) const;
	bool has_layer(const string& layer_name) const;
	const shared_ptr<Layer> layer_by_name(const string& layer_name) const;
	
	
protected:
	vector<string> layer_names_;
	vector<shared_ptr<Layer> > layers_;
	map<string, int> layer_names_index_;
	
	// Blob
	vector<shared_ptr<Blob> > blobs_;
	vector<string> blob_names_;
	map<string, int> blob_names_index_;
	
	// Host data
	vector<vector<Blob*> > bottom_vecs_;
	vector<vector<Blob*> > top_vecs_;
	vector<vector<int> > bottom_id_vecs_;
	vector<vector<int> > top_id_vecs_;
		
	// param
	vector<vector<string>> _param;
};

}