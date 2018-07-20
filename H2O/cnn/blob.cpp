#include <climits>
#include <vector>
#include "cnn/blob.hpp"

namespace H2O {

void Blob::Reshape(const int channels, const int height, const int width) 
{
	vector<int> shape(3);
	shape[0] = channels;
	shape[1] = height;
	shape[2] = width;
	Reshape(shape);
}

void Blob::Reshape(const vector<int>& shape) 
{
	count_ = 1;
	shape_.resize(shape.size());
	
	for (int i = 0; i < shape.size(); ++i) 
	{
		count_ *= shape[i];
		shape_[i] = shape[i];
	}
	if (count_ > capacity_) {
		capacity_ = count_;
		data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	}
}


void Blob::ReshapeLike(const Blob& other) {
	Reshape(other.shape());
}

Blob::Blob(const int channels, const int height, const int width) 
	: capacity_(0) 
{
	Reshape(channels, height, width);
}

Blob::Blob(const vector<int>& shape)
	// capacity_ must be initialized before calling Reshape
	: capacity_(0) 
{
	Reshape(shape);
}


const Dtype* Blob::cpu_data() const 
{
	if(!(data_))
	{
		fprintf(stderr, "cpu_data failed!\n");
		exit(0);
	}
	return (const Dtype*)data_->cpu_data();
}

void Blob::set_cpu_data(Dtype* data) 
{
	size_t size = count_ * sizeof(Dtype);
	if (data_->size() != size) {
		data_.reset(new SyncedMemory(size));
	}
	data_->set_cpu_data(data);
}

Dtype* Blob::mutable_cpu_data() 
{
	if(!(data_))
	{
		fprintf(stderr, "mutable_cpu_data failed!\n");
		exit(0);
	}
	return static_cast<Dtype*>(data_->mutable_cpu_data());
}

void Blob::ShareData(const Blob& other) 
{
	if(count_ != other.count())
	{
		fprintf(stderr, "ShareData failed!\n");
		exit(0);
	}
	data_ = other.data();
}


}
