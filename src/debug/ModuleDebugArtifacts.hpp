#pragma once
// Header-only instrumentation: standalone regression sources still link without
// the debug runner. No artifact is created unless a runner installs a case scope.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>
#include <opencv2/imgcodecs.hpp>

namespace cgmli { namespace debug {
inline thread_local std::filesystem::path caseDirectory;
inline std::string csvQuote(const std::string& value) {
    std::string result="\"";
    for(char c:value) { result+=c; if(c=='\"') result+='\"'; }
    return result+'\"';
}
inline std::ofstream artifactFile(const std::string& name) {
    if(caseDirectory.empty()) return {};
    const auto path=std::filesystem::u8path(name);
    if(path.has_parent_path() || path.filename()!=path || name.empty()) throw std::invalid_argument("Artifact name must be a filename");
    std::ofstream file(caseDirectory/path,std::ios::binary);
    if(!file) throw std::runtime_error("Cannot create debug artifact: "+name);
    file<<std::setprecision(17); return file;
}
/// Upper-bound error metric. A failed/nonfinite measurement fails the case even
/// when no runner is active; evidence is appended BEFORE raising the failure.
inline void recordError(const std::string& name,double error,double limit) {
    const bool passed=std::isfinite(error) && std::isfinite(limit) && error>=0 && error<=limit;
    if(!caseDirectory.empty()) {
        const auto path=caseDirectory/"metrics.csv";
        const bool header=!std::filesystem::exists(path);
        std::ofstream file(path,std::ios::binary|std::ios::app);
        if(!file) throw std::runtime_error("Cannot append module metrics");
        if(header) file<<"metric,error,limit,passed\n";
        file<<std::setprecision(17)<<csvQuote(name)<<','<<error<<','<<limit<<','<<int(passed)<<'\n';
        file.close(); if(!file) throw std::runtime_error("Module metrics write failed");
    }
    if(!passed) throw std::runtime_error("Numeric metric failed: "+name);
}
/// Requirement-level mixed tolerance: an absolute floor handles zero/cancelling
/// derivatives; the relative term is scaled by the ACTUAL reference norm, not 1+norm.
inline void recordDerivative(const std::string& name,double error,double referenceNorm,
    double relativeTolerance=1e-6,double absoluteTolerance=1e-7) {
    if(!std::isfinite(referenceNorm) || referenceNorm<0 || !std::isfinite(relativeTolerance)
        || relativeTolerance<=0 || !std::isfinite(absoluteTolerance) || absoluteTolerance<0)
        throw std::invalid_argument("Invalid derivative comparison policy");
    const double limit=absoluteTolerance+relativeTolerance*referenceNorm;
    if(!caseDirectory.empty()) {
        const auto path=caseDirectory/"derivative_checks.csv";
        const bool header=!std::filesystem::exists(path);
        std::ofstream file(path,std::ios::binary|std::ios::app);
        if(!file) throw std::runtime_error("Cannot append derivative checks");
        if(header) file<<"metric,absolute_error,reference_norm,absolute_floor,relative_tolerance,limit\n";
        file<<std::setprecision(17)<<csvQuote(name)<<','<<error<<','<<referenceNorm<<','
            <<absoluteTolerance<<','<<relativeTolerance<<','<<limit<<'\n';
        file.close(); if(!file) throw std::runtime_error("Derivative checks write failed");
    }
    recordError(name,error,limit);
}
inline void saveImage(const std::string& name,const cv::Mat& image) {
    if(caseDirectory.empty()) return;
    std::vector<uchar> bytes;
    if(image.empty() || !cv::imencode(".png",image,bytes)) throw std::runtime_error("Debug image encoding failed");
    auto file=artifactFile(name); file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
    file.close(); if(!file) throw std::runtime_error("Debug image write failed");
}
} }
