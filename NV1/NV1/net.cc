#include "NV1/net.h"
namespace NV1{

int Net::ncnn_param(const char* protopath)
{
	FILE* fp = fopen(protopath, "rb");
	CHECK(fp) << protopath << " fopen failed.";
	
    int magic = 0;
    fscanf(fp, "%d", &magic);
	CHECK(magic == 7767517) << "param is too old, please regenerate\n";

	// parse
    int layer_count = 0;
    int blob_count = 0;
    fscanf(fp, "%d %d", &layer_count, &blob_count);
	
	map<string, int> blob_name_to_idx;
	set<string> available_blobs;
	//vector<string> blob_names_;
	
	bottom_vecs_.resize(layer_count);
	top_vecs_.resize(layer_count);

	int layer_id = 0;
	while (!feof(fp))
    {
        int nscan = 0;

        char layer_type[257];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%256s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        { continue;	}
		
		for (int i=0; i<bottom_count; i++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            { continue;	}

			CHECK(available_blobs.find(blob_name) != available_blobs.end())
				<< "Unknown bottom blob -->> " << blob_name;

			const int blob_id = blob_name_to_idx[blob_name];
			bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
			available_blobs.erase(blob_name);
		}
		
		for (int i=0; i<top_count; i++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            { continue;	}
			
			CHECK(blob_name_to_idx.find(blob_name) == blob_name_to_idx.end())
				<< blob_name << " produced by multiple sources.";

			// Normal output.
			shared_ptr<Blob> blob_pointer(new Blob());
			const int blob_id = blobs_.size();
			blobs_.push_back(blob_pointer);
							
			blob_name_to_idx[blob_name] = blob_id;
			top_vecs_[layer_id].push_back(blob_pointer.get());		
			available_blobs.insert(blob_name);
		}
		
		// AppendParam
		char param[300];
		fgets(param, 300, fp);
		string paramstr = param;
		SetParam(paramstr);
		
		// Layers create
		layers_.push_back(LayerRegistry::CreateLayer(layer_name, layer_type, _param[layer_id]));
		layers_[layer_id]->xSetUp(bottom_vecs_[layer_id], top_vecs_[layer_id], this);
		
		layer_id++;
	}
	
	// Input <--> Output
	net_input_blobs_.push_back(blobs_[0].get());
	for (set<string>::iterator it = available_blobs.begin(); it != available_blobs.end(); ++it) 
	{
		LOG(INFO) << "This network produces output " << *it;
		net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
	}
	
	fclose(fp);
    return 0;
}

int Net::number_param()
{
	vector<vector<int> > NumInfo;	
#ifdef   USE_ModelH
	GetNumber_Net(NumInfo);
#endif
	CHECK(NumInfo.size());

#if 0
	for(int i=0; i<NumInfo.size(); i++)
	{
		for(int j=0; j<NumInfo[i].size(); j++)
			std::cout << NumInfo[i][j] << " ";
		
		std::cout << std::endl;
	}
#endif

	int layer_count = NumInfo.size();
	map<int, int> blob_name_to_idx;
	set<int> available_blobs;
	
	bottom_vecs_.resize(layer_count);
	top_vecs_.resize(layer_count);
	
	for(int i=0; i<layer_count; i++)
	{
		int bot_count = NumInfo[i][2];
        int top_count = NumInfo[i][3];
		
		for (int m=0; m<bot_count; m++)
        {
			int blob_name = NumInfo[i][4+m];
			CHECK(available_blobs.find(blob_name) != available_blobs.end())
				<< "Unknown bottom blob -->> " << blob_name;
				
			const int blob_id = blob_name_to_idx[blob_name];
			bottom_vecs_[i].push_back(blobs_[blob_id].get());
			available_blobs.erase(blob_name);
		}
		
		for (int m=0; m<top_count; m++)
        {
			int blob_name = NumInfo[i][4+bot_count+m];
			CHECK(blob_name_to_idx.find(blob_name) == blob_name_to_idx.end())
				<< blob_name << " produced by multiple sources.";

			// Normal output.
			shared_ptr<Blob> blob_pointer(new Blob());
			const int blob_id = blobs_.size();
			blobs_.push_back(blob_pointer);
							
			blob_name_to_idx[blob_name] = blob_id;
			top_vecs_[i].push_back(blob_pointer.get());		
			available_blobs.insert(blob_name);
		}
	}
	
	return 0;
}

int Net::Yolo_model(FILE* fp)
{
	int ret = 0;
	
	int major, minor, revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
	size_t seen;
	if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(&seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        seen = iseen;
    }
	
	int transpose = (major > 1000) || (minor > 1000);
	//fprintf(stderr, "%d %d %d %d\n", major, minor, revision, transpose);
	
	int num_source_layers = layers_.size();
	for (int i = 0; i < num_source_layers; ++i) {
		
	string layer_type = layers_[i]->type();
	if(layer_type == "Convolution")
	{
		if(layers_[i]->blobs_.size() == 5)
		{
			fread(layers_[i]->blobs_[1]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[1]->count(), fp);
			fread(layers_[i]->blobs_[2]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[2]->count(), fp);
			fread(layers_[i]->blobs_[3]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[3]->count(), fp);
			fread(layers_[i]->blobs_[4]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[4]->count(), fp);
			fread(layers_[i]->blobs_[0]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[0]->count(), fp);

		}
		else if(layers_[i]->blobs_.size() == 2)
		{
			fread(layers_[i]->blobs_[1]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[1]->count(), fp);
			fread(layers_[i]->blobs_[0]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[0]->count(), fp);
		}
	}
	}

	return ret;
}

int Net::ncnn_model(FILE* binfp)
{
	int nread;
	
	int num_source_layers = layers_.size();
	for (int i = 0; i < num_source_layers; ++i) 
	{
		string layer_type = layers_[i]->type();
		LOG(INFO) << layer_type << " " << layers_[i]->blobs_.size();
		
		if(layer_type == "Convolution" || layer_type == "ConvolutionDepthWise" || layer_type == "InnerProduct")
		{
			int flag_struct = 0;
			nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
			CHECK(nread == 1);
			LOG(INFO) << "Flag_struct: " << flag_struct;
			
			for (int j=0; j<layers_[i]->blobs_.size(); ++j) 
			{
				LOG(INFO) << layers_[i]->blobs_[j]->count();
				int count = layers_[i]->blobs_[j]->count();
				float *data = layers_[i]->blobs_[j]->mutable_cpu_data();
				
				nread = fread(data, count * sizeof(float), 1, binfp);
				CHECK(nread == 1);
			}
		}
	}
}

int Net::load_model(const char* modelpath)
{
	FILE* fp = fopen(modelpath, "rb");
	if (!fp)
	{
		fprintf(stderr, "fopen %s failed\n", modelpath);
		return -1;
	}
	
	//int ret = Yolo_model(fp);
	int ret = ncnn_model(fp);
	fclose(fp);

	return ret;
}

void Net::SetParam(string paramstr)
{
	int npos = 0;
	vector<string> setparam;
	
	for(int i=0; i<30; i++){
		
	int s_index = paramstr.find('=', npos);
	npos = s_index;
	if(s_index >= 0)
	{
		int e_index = paramstr.find(' ', npos);
		npos = e_index;
		if(e_index >= 0)
		{
			string name = paramstr.substr(s_index+1, e_index-1-s_index);
			setparam.push_back(name);
		}
		else
		{
			string name = paramstr.substr(s_index+1, paramstr.size()-2-s_index);
			setparam.push_back(name);
		}
	}
	else  break;
	
	}
	_param.push_back(setparam);
}

void Net::Run() 
{
	// 前向网络 <--> 预测
	for (int i = 0; i < layers_.size(); ++i) 
	{
		layers_[i]->Run(bottom_vecs_[i], top_vecs_[i]);
	}
}

}