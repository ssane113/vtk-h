#ifndef vtkh_loggin_h
#define vtkh_loggin_h

#include <fstream>
#include <stack>
#include <sstream>

namespace vtkh {

class Logger 
{
public:
  ~Logger();
  static Logger *get_instance();
  void write(const int level, const std::string &message, const char *file, int line);
  std::ofstream & get_stream();
protected:
  Logger();
  Logger(Logger const &);
  std::ofstream m_stream;
  static class Logger* m_instance;
};

class DataLogger 
{
public:
  ~DataLogger();
  static DataLogger *GetInstance();
  void OpenLogEntry(const std::string &entryName);
  void CloseLogEntry(const double &entryTime);

  template<typename T>
  void AddLogData(const std::string key, const T &value)
  {
    this->Stream<<key<<" "<<value<<"\n";
  }

  std::stringstream& GetStream();
  void WriteLog();
protected:
  DataLogger();
  DataLogger(DataLogger const &);
  std::stringstream Stream;
  static class DataLogger* Instance;
  std::stack<std::string> Entries;
};

#define VTKH_INFO(msg) vtkh::Logger::get_instance()->get_stream() <<"<Info>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define VTKH_WARN(msg) vtkh::Logger::get_instance()->get_stream() <<"<Warn>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;
#define VTKH_ERROR(msg) vtkh::Logger::get_instance()->get_stream() <<"<Error>\n" \
  <<"  message: "<< msg <<"\n  file: " <<__FILE__<<"\n  line:  "<<__LINE__<<std::endl;

} // namespace rover

#endif
