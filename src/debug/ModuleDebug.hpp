#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cgmli { namespace debug {
inline constexpr const char* validationPolicy="2026-11-05-v16";
struct ModuleCase {
    std::string module,name;
    std::function<void()> execute;
};
void registerExtractionCases(std::vector<ModuleCase>& cases);
void registerOrbCases(std::vector<ModuleCase>& cases);
void registerPoseCases(std::vector<ModuleCase>& cases);
void registerAssociationCases(std::vector<ModuleCase>& cases);
void registerBackendCases(std::vector<ModuleCase>& cases);
void registerPipelineCases(std::vector<ModuleCase>& cases);
int runModuleDebug(int argc,char** argv);
} }
