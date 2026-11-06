"""Generate the independent VS debug target from current dependency settings; no build."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "CGMLI-SLAM/CGMLI-SLAM.vcxproj"
GUID = "{6458ECBD-927C-4BDB-BCAC-4B2C8E1D253C}"
NS = "http://schemas.microsoft.com/developer/msbuild/2003"
ET.register_namespace("", NS)
def tag(name): return f"{{{NS}}}{name}"
TESTS = ["src/visual/features/tests/OrbCameraAwareRegression.cpp", "src/frontend/tests/FrontendOrbRegression.cpp",
         "src/frontend/tests/FrontendPoseRegression.cpp", "src/frontend/tests/FrontendBackendRegression.cpp",
         "src/backend/tests/ManagedWindowRegression.cpp", "src/pipeline/tests/NativeSlamRegression.cpp"]
DEBUG = ["src/debug/ModuleDebugMain.cpp", "src/debug/ModuleDebug.cpp"]
HEADERS = ["src/debug/ModuleDebug.hpp", "src/debug/ModuleDebugArtifacts.hpp"]
def relative(path): return "..\\" + path.replace("/", "\\")

tree = ET.parse(PROJECT); root = tree.getroot()
for name, value in (("ProjectGuid", GUID), ("RootNamespace", "CGMLIModuleDebug")):
    root.find(f"{tag('PropertyGroup')}[@Label='Globals']/{tag(name)}").text = value
settings = ET.Element(tag("PropertyGroup"))
for key, value in (("TargetName", "CGMLI-ModuleDebug"),
                   ("OutDir", "$(ProjectDir)$(Platform)\\$(Configuration)\\"),
                   ("IntDir", "$(ProjectDir)module_debug\\$(Platform)\\$(Configuration)\\obj\\")):
    ET.SubElement(settings, tag(key)).text = value
# Set artifact paths before Microsoft.Cpp.props expands intermediate paths.
index = next(i for i, child in enumerate(root) if child.tag == tag("Import") and child.get("Project", "").endswith("Microsoft.Cpp.props"))
root.insert(index, settings)
test_paths = {relative(path) for path in TESTS}
for group in root.findall(tag("ItemGroup")):
    for child in list(group):
        include = child.get("Include", "")
        if (child.tag == tag("ClCompile") and include.endswith("\\app\\offline_main.cpp")) or (child.tag == tag("None") and include in test_paths):
            group.remove(child)
for compile_settings in root.findall(f"{tag('ItemDefinitionGroup')}/{tag('ClCompile')}"):
    definitions = compile_settings.find(tag("PreprocessorDefinitions"))
    if definitions is None: definitions = ET.SubElement(compile_settings, tag("PreprocessorDefinitions"))
    definitions.text = "CGMLI_EMBED_MODULE_TESTS;" + (definitions.text or "%(PreprocessorDefinitions)")
    options = compile_settings.find(tag("AdditionalOptions"))
    if options is None: options = ET.SubElement(compile_settings, tag("AdditionalOptions"))
    options.text = "/utf-8 " + (options.text or "%(AdditionalOptions)")
items = ET.SubElement(root, tag("ItemGroup"))
for path in TESTS + DEBUG: ET.SubElement(items, tag("ClCompile"), Include=relative(path))
for path in HEADERS: ET.SubElement(items, tag("ClInclude"), Include=relative(path))
ET.indent(tree, space="  ")
tree.write(PROJECT.with_name("CGMLI-ModuleDebug.vcxproj"), encoding="utf-8", xml_declaration=True)

filters = ET.parse(PROJECT.with_suffix(".vcxproj.filters")); fr = filters.getroot()
for group in fr.findall(tag("ItemGroup")):
    for child in list(group):
        include = child.get("Include", "")
        if include.endswith("\\app\\offline_main.cpp") or include in test_paths: group.remove(child)
fg = ET.SubElement(fr, tag("ItemGroup")); folder = ET.SubElement(fg, tag("Filter"), Include="Module Debug")
ET.SubElement(folder, tag("UniqueIdentifier")).text = "{A8644C43-695B-4DE2-B3CF-95BD37A18E0D}"
group = ET.SubElement(fr, tag("ItemGroup"))
for path in TESTS + DEBUG + HEADERS:
    item = ET.SubElement(group, tag("ClCompile") if path.endswith(".cpp") else tag("ClInclude"), Include=relative(path))
    ET.SubElement(item, tag("Filter")).text = "Module Debug"
ET.indent(filters, space="  ")
filters.write(PROJECT.with_name("CGMLI-ModuleDebug.vcxproj.filters"), encoding="utf-8", xml_declaration=True)

solution = PROJECT.with_suffix(".sln"); text = solution.read_text(encoding="utf-8-sig")
if GUID not in text:
    block = f'Project("{{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}}") = "CGMLI-ModuleDebug", "CGMLI-ModuleDebug.vcxproj", "{GUID}"\nEndProject\n'
    text = text.replace("\nGlobal\n", "\n" + block + "Global\n", 1)
    configurations = ""
    for config in ("Debug", "Release"):
        for platform, native in (("x64", "x64"), ("x86", "Win32")):
            configurations += f"\t\t{GUID}.{config}|{platform}.ActiveCfg = {config}|{native}\n"
    anchor = "\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\n"
    # Deliberately no Build.0: ordinary solution builds keep the production target.
    text = text.replace(anchor, anchor + configurations, 1)
    solution.write_text(text, encoding="utf-8-sig")
print("Generated CGMLI-ModuleDebug project/filters and registered it in the solution; not compiled.")
