#include "ModuleDebug.hpp"
#include <glog/logging.h>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {
int application(int argc,char** argv) {
    google::InitGoogleLogging(argv[0]); FLAGS_logtostderr=true;
    struct Shutdown { ~Shutdown() { google::ShutdownGoogleLogging(); } } shutdown;
    try { return cgmli::debug::runModuleDebug(argc,argv); }
    catch(const std::exception& error) { LOG(ERROR)<<"[ModuleDebug] "<<error.what(); return 2; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    std::vector<std::string> storage; storage.reserve(size_t(argc));
    for(int i=0;i<argc;++i) storage.push_back(std::filesystem::path(argv[i]).u8string());
    std::vector<char*> arguments; for(auto& value:storage) arguments.push_back(value.data());
    return application(argc,arguments.data());
}
#else
int main(int argc,char** argv) { return application(argc,argv); }
#endif
