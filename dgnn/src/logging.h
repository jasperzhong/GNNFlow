#ifndef DGNN_LOGGING_H_
#define DGNN_LOGGING_H_

#include <sstream>
#include <string>

namespace dgnn {

#define CUDA_CALL(func)                                              \
  {                                                                  \
    cudaError_t e = (func);                                          \
    if (e != cudaSuccess && e != cudaErrorCudartUnloading)           \
      throw thrust::system_error(e, thrust::cuda_category(), #func); \
  }

enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };

constexpr char LOG_LEVELS[] = "TDIWEF";

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* file, int line, LogLevel level);
  virtual ~LogMessage();

 private:
  const char* file_;
  int line_;
  LogLevel level_;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define CHECK(x) \
  if (!(x)) dgnn::LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_NOTNULL(x) CHECK((x) != nullptr)

#define LOG(level) dgnn::LogMessage(__FILE__, __LINE__, dgnn::LogLevel::level)
}  // namespace dgnn

#endif  // DGNN_LOGGING_H_
