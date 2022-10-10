#include "logging.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace gnnflow {
LogLevel ParseLogLevelStr(const char* env_var_val) {
  std::string min_log_level(env_var_val);
  std::transform(min_log_level.begin(), min_log_level.end(),
                 min_log_level.begin(), ::tolower);
  if (min_log_level == "trace") {
    return LogLevel::TRACE;
  } else if (min_log_level == "debug") {
    return LogLevel::DEBUG;
  } else if (min_log_level == "info") {
    return LogLevel::INFO;
  } else if (min_log_level == "warning") {
    return LogLevel::WARNING;
  } else if (min_log_level == "error") {
    return LogLevel::ERROR;
  } else if (min_log_level == "fatal") {
    return LogLevel::FATAL;
  } else {
    return LogLevel::WARNING;
  }
}

LogLevel MinLogLevelFromEnv() {
  const char* env_var_val = getenv("LOGLEVEL");
  if (env_var_val == nullptr) {
    return LogLevel::WARNING;
  }
  return ParseLogLevelStr(env_var_val);
}

LogMessage::LogMessage(const char* file, int line, LogLevel level)
    : file_(file), line_(line), level_(level) {}

LogMessage::~LogMessage() {
  bool use_cout = static_cast<int>(level_) <= static_cast<int>(LogLevel::INFO);
  std::ostream& os = use_cout ? std::cout : std::cerr;

  if (level_ >= MinLogLevelFromEnv()) {
    os << "[" << LOG_LEVELS[static_cast<int>(level_)] << "] " << file_ << ":"
       << line_ << ": " << str() << std::endl;
  }
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, LogLevel::FATAL) {}

LogMessageFatal::~LogMessageFatal() { std::abort(); }
};  // namespace gnnflow
