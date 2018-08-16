#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <map>
#include <set>
#include "nnet/syncedmem.h"

#define Dtype float

namespace NNET
{

#define DISABLE_COPY_AND_ASSIGN(classname) 	\
private:									\
  classname(const classname&);				\
  classname& operator=(const classname&)	

using std::shared_ptr;  
using std::set;
using std::map;
using std::string;
using std::vector;
using std::ostringstream;


class Blob {
public:
	Blob()
		: data_(), count_(0), capacity_(0) {}
	explicit Blob(const int channels, const int height = 1, const int width = 1);
	//~Blob() {LOG(INFO) << count_ << "aaa";}
	
	explicit Blob(const vector<int>& shape);

	void Reshape(const int channels, const int height = 1, const int width = 1);

	void Reshape(const vector<int>& shape);
	void ReshapeLike(const Blob& other);
	inline string shape_string() const 
	{
		ostringstream stream;
		for (int i = 0; i < shape_.size(); ++i) 
		{
			stream << shape_[i] << " ";
		}
		stream << "(" << count_ << ")";
		return stream.str();
	}
	inline const vector<int>& shape() const { return shape_; }


	//**************//
	inline int shape(int index) const {
		return shape_[CanonicalAxisIndex(index)];
	}
	inline int num_axes() const { return shape_.size(); }
	inline int count() const { return count_; }

	
	inline int count(int start_axis, int end_axis) const 
	{
		int count = 1;
		for (int i = start_axis; i < end_axis; ++i) {
		  count *= shape(i);
		}
		return count;
	}
	inline int count(int start_axis) const {
		return count(start_axis, num_axes());
	}

	inline int CanonicalAxisIndex(int axis_index) const 
	{
		if (axis_index < 0) {
			return axis_index + num_axes();
		}
		return axis_index;
	}

	// num channels height width
	inline int CC() const { return LegacyShape(0); }
	inline int HH() const { return LegacyShape(1); }
	inline int WW() const { return LegacyShape(2); }
	inline int LegacyShape(int index) const 
	{
		if (index >= num_axes() || index < -num_axes()) 
		{
			return 1;
		}
		return shape(index);
	}

	inline int offset(const int c = 0, const int h = 0,
		const int w = 0) const 
	{
		return (c * HH() + h) * WW() + w;
	}


	//************************//Copy from a source Blob.
	void CopyFrom(const Blob& source, bool copy_diff = false, bool reshape = false);

	inline Dtype data_at(const int c, const int h, const int w) const 
	{
		return cpu_data()[offset(c, h, w)];
	}

	inline const shared_ptr<SyncedMemory>& data() const
	{
		CHECK(data_);
		return data_;
	}

	const Dtype* cpu_data() const;
	
	void set_cpu_data(Dtype* data);
	
	Dtype* mutable_cpu_data();

	void ShareData(const Blob& other);


protected:
	shared_ptr<SyncedMemory> data_;
	vector<int> shape_;
	int count_;
	int capacity_;

DISABLE_COPY_AND_ASSIGN(Blob);
};

}