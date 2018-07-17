#include "cnn/net.hpp"

namespace cnn 
{

Net::Net()
{}

Net::~Net()
{
  ;
}

int Net::load_param(FILE* fp)
{
    int magic = 0;
    fscanf(fp, "%d", &magic);
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }
	// parse
    int layer_count = 0;
    int blob_count = 0;
    fscanf(fp, "%d %d", &layer_count, &blob_count);
	
	layer_names_.resize(layer_count);
	map<string, int> blob_name_to_idx;
	set<string> available_blobs;
	
	bottom_vecs_.resize(layer_count);
	top_vecs_.resize(layer_count);
	bottom_id_vecs_.resize(layer_count);
	top_id_vecs_.resize(layer_count);
	
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
        {
            continue;
        }
		
		layer_names_[layer_id] = layer_name;
		//fprintf(stderr, "Layer %s\n", layer_name);
		
		for (int i=0; i<bottom_count; i++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

			// AppendBottom 
			if (available_blobs.find(blob_name) == available_blobs.end()) 
			{
				fprintf(stderr, "Unknown bottom blob %s\n", blob_name);
				return -1;
			}
			const int blob_id = blob_name_to_idx[blob_name];
			bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
			bottom_id_vecs_[layer_id].push_back(blob_id);
			//available_blobs.erase(blob_name);
		}
		
		for (int i=0; i<top_count; i++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }
			
			// AppendTop
			if(blob_name_to_idx.find(blob_name) != blob_name_to_idx.end()) 
			{
				fprintf(stderr, "Top blob %s produced by multiple sources.\n", blob_name);
				return -1;
			}
			else
			{
				// Normal output.
				shared_ptr<Blob> blob_pointer(new Blob());
				const int blob_id = blobs_.size();
				blobs_.push_back(blob_pointer);
				blob_names_.push_back(blob_name);
								
				blob_name_to_idx[blob_name] = blob_id;
				top_id_vecs_[layer_id].push_back(blob_id);
				top_vecs_[layer_id].push_back(blob_pointer.get());
			}
			
			available_blobs.insert(blob_name);
		}
		
		// AppendParam
		char param[300];
		fgets(param, 300, fp);
		string paramstr = param;
		SetParam(paramstr);
		
		// Layers create
		layers_.push_back(LayerRegistry::CreateLayer(layer_name, layer_type, _param[layer_id]));
		layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
		
		layer_id++;
	}
	
	for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) 
	{
		blob_names_index_[blob_names_[blob_id]] = blob_id;
	}
	for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) 
	{
		layer_names_index_[layer_names_[layer_id]] = layer_id;
	}
	
    return 0;
}

int Net::load_param(const char* protopath)
{
	FILE* fp = fopen(protopath, "rb");
	if (!fp)
	{
		fprintf(stderr, "fopen %s failed\n", protopath);
		return -1;
	}
	
	int ret = load_param(fp);
	fclose(fp);

	return ret;
}

int Net::load_model(FILE* fp)
{
  // load file
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
	//fprintf(stderr, ">>%s\n", layer_type.c_str());
	if(layer_type == "Convolution")
	{
		//fprintf(stderr, "Conv[%d] blob.size: %ld\n", i, layers_[i]->blobs_.size());
		if(layers_[i]->blobs_.size() == 5)
		{
			fread(layers_[i]->blobs_[1]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[1]->count(), fp);
			fread(layers_[i]->blobs_[2]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[2]->count(), fp);
			fread(layers_[i]->blobs_[3]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[3]->count(), fp);
			fread(layers_[i]->blobs_[4]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[4]->count(), fp);
			fread(layers_[i]->blobs_[0]->mutable_cpu_data(), sizeof(float), layers_[i]->blobs_[0]->count(), fp);
			//fprintf(stderr, ">> %d %d %d %d %d\n",
			//layers_[i]->blobs_[0]->count(), layers_[i]->blobs_[1]->count(),
			//layers_[i]->blobs_[2]->count(), layers_[i]->blobs_[3]->count(),
			//layers_[i]->blobs_[4]->count());
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

int Net::load_model(const char* modelpath)
{
	FILE* fp = fopen(modelpath, "rb");
	if (!fp)
	{
		fprintf(stderr, "fopen %s failed\n", modelpath);
		return -1;
	}
	
	int ret = load_model(fp);
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

void Net::Forward() 
{
	for (int i = 0; i < layers_.size(); ++i) 
	{
		layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
	}
}


bool Net::has_blob(const string& blob_name) const {
	return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

const shared_ptr<Blob> Net::blob_by_name(const string& blob_name) const 
{
	shared_ptr<Blob> blob_ptr;
	if (has_blob(blob_name)) {
		blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
	}
	else {
		blob_ptr.reset((Blob*)(NULL));
		fprintf(stderr, "Unknown blob name %s\n", blob_name.c_str());
	}
	return blob_ptr;
}

bool Net::has_layer(const string& layer_name) const {
	return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

const shared_ptr<Layer> Net::layer_by_name(const string& layer_name) const 
{
	shared_ptr<Layer> layer_ptr;
	if (has_layer(layer_name)) {
		layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
	}
	else {
		layer_ptr.reset((Layer*)(NULL));
		fprintf(stderr, "Unknown layer name %s\n", layer_name.c_str());
	}
	return layer_ptr;
}

}