#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

namespace NV1{
#define LOG_ERROR 		LOG_INFO
#define LOG_WARNING 	LOG_INFO
#define LOG(severity) 	LOG_##severity.stream()
#define LOG_INFO 		NV1::LogMessage(__FILE__, __LINE__)
#define LOG_FATAL 		NV1::LogMessageFatal(__FILE__, __LINE__)

#define CHECK(x)	\
if (!(x))  			\
NV1::LogMessageFatal(__FILE__, __LINE__).stream() << "Check Failed: " #x << ' '\


#define CHECK_NOTNULL(x) \
((x) == NULL ? caffe::LogMessageFatal(__FILE__, __LINE__).stream() 	\
<< "Check  notnull: "  #x << ' ', (x) : (x)) 						\


//***********************************************//	  
struct Error : public std::runtime_error 
{
	explicit Error(const std::string &s) 
		: std::runtime_error(s) {}
};

class LogMessage 
{
public:
  LogMessage(const char* file, int line) :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
	{	log_stream_ << "[" << file << "]" << line << ": "; 	 }
  
	~LogMessage() { log_stream_ << '\n'; }
	std::ostream& stream() { return log_stream_; }

protected:
	std::ostream& log_stream_;
};

class LogMessageFatal 
{
public:
	LogMessageFatal(const char* file, int line)
	{	log_stream_ << "[" << file << "]" << line << ": ";	 }
	
#if defined(_MSC_VER) && _MSC_VER < 1900
	~LogMessageFatal() {
#else
	~LogMessageFatal() noexcept(false) 
	{
#endif
		LOG(ERROR) << log_stream_.str();
		throw Error(log_stream_.str());
	}
	std::ostringstream &stream() { return log_stream_; }

private:
  std::ostringstream log_stream_;
};
//***********************************************//	

}