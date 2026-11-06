#include "ModuleDebug.hpp"
#include "ModuleDebugArtifacts.hpp"
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>

namespace cgmli { namespace debug {
namespace {
namespace fs=std::filesystem;
std::string html(const std::string& value) {
    std::string result;
    for(char c:value) switch(c) {
        case '&': result+="&amp;"; break; case '<': result+="&lt;"; break;
        case '>': result+="&gt;"; break; case '"': result+="&quot;"; break;
        default: result+=c;
    }
    return result;
}
struct Outcome { std::string module,name,message; fs::path folder; bool passed=false; double ms=0; };
class FileSink final : public google::LogSink {
public:
    explicit FileSink(const fs::path& path):file_(path,std::ios::binary) {
        if(!file_) throw std::runtime_error("Cannot create module debug log"); google::AddLogSink(this);
    }
    ~FileSink() override { google::RemoveLogSink(this); file_.flush(); }
    void send(google::LogSeverity severity,const char*,const char* file,int line,const std::tm* time,const char* message,size_t length) override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_<<google::LogSink::ToString(severity,file,line,time,message,length)<<'\n';
        if(severity>=google::GLOG_WARNING) file_.flush();
    }
private: std::ofstream file_; std::mutex mutex_;
};
fs::path defaultRoot(const char* executable) {
    auto directory=fs::absolute(fs::u8path(executable)).parent_path();
    for(auto candidate=directory;!candidate.empty();) {
        if(fs::is_directory(candidate/"src"/"backend") && fs::is_directory(candidate/"config")) return candidate/"debug_results";
        auto parent=candidate.parent_path(); if(parent==candidate) break; candidate=parent;
    }
    return directory/"debug_results";
}
fs::path createRun(const fs::path& root) {
    fs::create_directories(root); const auto stamp=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); std::tm calendar{};
#ifdef _WIN32
    localtime_s(&calendar,&stamp);
#else
    localtime_r(&stamp,&calendar);
#endif
    std::ostringstream name; name<<"run_"<<std::put_time(&calendar,"%Y%m%d_%H%M%S");
    for(size_t n=0;;++n) { const auto run=root/(name.str()+"_"+std::to_string(n)); if(fs::create_directory(run)) return run; }
}
void writeReport(const fs::path& run,const std::vector<Outcome>& outcomes,size_t selected) {
    std::ofstream csv(run/"cases.csv",std::ios::binary),page(run/"report.html",std::ios::binary);
    if(!csv || !page) throw std::runtime_error("Cannot create module report");
    csv<<"module,case,status,elapsed_ms,message,data_scope\n";
    page<<u8"<!doctype html><meta charset='utf-8'><title>CGMLI module debug</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;background:#f7f8fa;color:#182333}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}a{color:#155db5}.pass{color:#147347}.fail{color:#b42318}code{font-size:13px}img{max-width:100%;margin:10px 0}summary{cursor:pointer}</style>"
        <<u8"<h1>模块正确性调试报告</h1><p>受控合成用例；PASS 仅覆盖该用例检查的条件，不表示真实数据集精度或完整 SLAM 已验证。</p>"
        <<"<p>Validation policy: <code>"<<validationPolicy<<"</code>; build: "<<__DATE__<<' '<<__TIME__<<"</p>"
        <<"<p><a href='cases.csv'>cases.csv</a> | <a href='module.log'>glog</a></p><label><input id='failOnly' type='checkbox' onchange=\"document.querySelectorAll('tr[data-pass]').forEach(r=>r.hidden=this.checked&&r.dataset.pass==='1')\">仅显示失败用例</label><table><tr><th>Module / case</th><th>Status</th><th>ms</th><th>Evidence</th></tr>";
    size_t failures=0;
    for(const auto& outcome:outcomes) {
        failures+=!outcome.passed;
        csv<<csvQuote(outcome.module)<<','<<csvQuote(outcome.name)<<','<<(outcome.passed?"PASS":"FAIL")<<','<<outcome.ms<<','<<csvQuote(outcome.message)<<",synthetic\n";
        const auto relative=outcome.folder.filename().u8string();
        page<<"<tr data-pass='"<<int(outcome.passed)<<"'><td>"<<html(outcome.module)<<"<br><code>"<<html(outcome.name)<<"</code></td><td class='"<<(outcome.passed?"pass":"fail")<<"'>"<<(outcome.passed?"PASS":"FAIL")<<"</td><td>"<<outcome.ms<<"</td><td>"<<html(outcome.message);
        for(const auto& entry:fs::directory_iterator(outcome.folder)) if(entry.is_regular_file()) {
            const auto name=entry.path().filename().u8string(); const auto link=relative+"/"+name;
            if(entry.path().extension()==".png") page<<"<details><summary>"<<html(name)<<"</summary><a href='"<<html(link)<<"'><img loading='lazy' src='"<<html(link)<<"'></a></details>";
            else page<<"<br><a href='"<<html(link)<<"'>"<<html(name)<<"</a>";
        }
        page<<"</td></tr>";
    }
    page<<"</table><p>Selected: "<<selected<<"; completed: "<<outcomes.size()<<"; failures: "<<failures
        <<"; state: "<<(outcomes.size()==selected?"COMPLETE":"INCOMPLETE / RUNNING")<<". Compiled runtime checks, not static source checks.</p>";
    csv.close(); page.close(); if(!csv || !page) throw std::runtime_error("Module report write failed");
    std::ofstream status(run/"status.json",std::ios::binary);
    status<<"{\"scope\":\"synthetic\",\"validation_policy\":\""<<validationPolicy<<"\",\"selected\":"<<selected<<",\"completed\":"<<outcomes.size()
        <<",\"failed\":"<<failures<<",\"complete\":"<<(outcomes.size()==selected?"true":"false")<<"}\n";
    status.close(); if(!status) throw std::runtime_error("Module status write failed");
}
}
int runModuleDebug(int argc,char** argv) {
    std::vector<ModuleCase> cases;
    registerExtractionCases(cases); registerOrbCases(cases); registerPoseCases(cases);
    registerAssociationCases(cases); registerBackendCases(cases); registerPipelineCases(cases);
    if(argc==2 && std::string(argv[1])=="--list") {
        for(const auto& test:cases) std::cout<<test.module<<" / "<<test.name<<'\n'; return 0;
    }
    if(argc<3 || std::string(argv[1])!="--module") {
        std::cerr<<"Usage: CGMLI-ModuleDebug --list\n       CGMLI-ModuleDebug --module MODULE [output_root] [--case substring]\n"; return 2;
    }
    const std::string module=argv[2]; std::string filter; auto root=defaultRoot(argv[0]); bool suppliedRoot=false;
    for(int i=3;i<argc;++i) {
        const std::string argument=argv[i];
        if(argument=="--case" && i+1<argc && filter.empty()) filter=argv[++i];
        else if(argument.rfind("--",0)!=0 && !suppliedRoot) { root=fs::u8path(argument); suppliedRoot=true; }
        else throw std::invalid_argument("Unknown or incomplete debug argument: "+argument);
    }
    std::vector<ModuleCase> selected;
    for(const auto& test:cases) if((module=="all" || test.module==module) && (filter.empty() || test.name.find(filter)!=std::string::npos)) selected.push_back(test);
    if(selected.empty()) throw std::invalid_argument("No matching module/case; use --list");
    root=fs::absolute(root); const auto run=createRun(root); FileSink log(run/"module.log");
    std::vector<Outcome> outcomes;
    LOG(INFO)<<"[ModuleDebug] synthetic controlled checks; policy="<<validationPolicy<<" module="<<module<<" cases="<<selected.size()<<" output="<<run.u8string();
    std::ofstream arguments(run/"arguments.txt",std::ios::binary); for(int i=0;i<argc;++i) arguments<<i<<": "<<argv[i]<<'\n';
    arguments.close(); if(!arguments) throw std::runtime_error("Cannot write module arguments");
    writeReport(run,outcomes,selected.size());
    std::ofstream latest(root/"latest.txt",std::ios::binary); latest<<(run/"report.html").u8string()<<'\n'; latest.close();
    if(!latest) throw std::runtime_error("Cannot write debug latest pointer");
    for(const auto& test:selected) {
        Outcome outcome; outcome.module=test.module; outcome.name=test.name; outcome.folder=run/(test.module+"_"+test.name);
        fs::create_directory(outcome.folder); caseDirectory=outcome.folder;
        const auto started=std::chrono::steady_clock::now();
        LOG(INFO)<<"[ModuleDebug] BEGIN "<<test.module<<'/'<<test.name;
        try { test.execute(); outcome.passed=true; outcome.message="covered assertions passed"; }
        catch(const std::exception& error) { outcome.message=error.what(); }
        catch(...) { outcome.message="non-standard exception"; }
        caseDirectory.clear(); outcome.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        LOG(INFO)<<"[ModuleDebug] "<<(outcome.passed?"PASS ":"FAIL ")<<test.module<<'/'<<test.name<<" ms="<<outcome.ms<<" reason="<<outcome.message;
        outcomes.push_back(std::move(outcome)); writeReport(run,outcomes,selected.size());
    }
    const auto failures=std::count_if(outcomes.begin(),outcomes.end(),[](const auto& outcome){return !outcome.passed;});
    LOG(INFO)<<"[ModuleDebug] completed="<<outcomes.size()<<" failures="<<failures<<" report="<<(run/"report.html").u8string();
    return failures?1:0;
}
} }
