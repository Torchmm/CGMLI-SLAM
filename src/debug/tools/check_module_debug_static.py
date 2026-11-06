"""Inspect the module-debug target and evidence plumbing; never build or run C++."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
NS = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
checks = []
def check(ok, name): checks.append({"name": name, "passed": bool(ok)})
project = ET.parse(ROOT / "CGMLI-SLAM/CGMLI-ModuleDebug.vcxproj")
production = ET.parse(ROOT / "CGMLI-SLAM/CGMLI-SLAM.vcxproj")
compiled = [n.get("Include") for n in project.findall(".//m:ClCompile[@Include]", NS)]
check(not any(p.endswith("offline_main.cpp") for p in compiled), "replay main excluded")
check(any(p.endswith("ModuleDebugMain.cpp") for p in compiled), "dedicated main registered")
check(not any("\\tests\\" in n.get("Include") for n in production.findall(".//m:ClCompile[@Include]", NS)), "production build remains independent")
for node in project.findall(".//m:ItemDefinitionGroup/m:ClCompile", NS):
    check("CGMLI_EMBED_MODULE_TESTS" in node.find("m:PreprocessorDefinitions", NS).text, "embedded main suppression configured")
modules = set(); names = set(); suites = 0
strip = r'/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
for include in compiled:
    path = (ROOT / "CGMLI-SLAM" / include.replace("\\", "/")).resolve()
    check(path.is_file(), "compiled source exists: " + path.name)
    if "/tests/" not in path.as_posix(): continue
    suites += 1; text = path.read_text(encoding="utf-8-sig")
    check("#ifdef CGMLI_EMBED_MODULE_TESTS" in text and "#else\nint main()" in text, path.name + ": standalone main guarded")
    # Broad using-directives can expose several unrelated detail namespaces.
    clean = re.sub(strip, " ", text, flags=re.S)
    check(not re.search(r"(?<![\w:])detail\s*::", clean), path.name + ": detail references explicitly qualified")
    for module, name in re.findall(r'cases\.push_back\(\{"(\w+)","(\w+)"', text):
        check((module, name) not in names, "unique case: " + module + "/" + name)
        check(bool(re.search(r"\b(?:void|bool|int)\s+" + ("cameraVariant" if name.startswith("cameraVariant_") else name) + r"\s*\(", text)), "registered function defined: " + name)
        names.add((module, name)); modules.add(module)
check(suites == 6, "six existing suites embedded")
check(modules == {"extraction", "matching", "initialization", "tracking", "backend", "jacobian", "marginalization"}, "all seven requested modules registered")
filters = ET.parse(ROOT / "CGMLI-SLAM/CGMLI-ModuleDebug.vcxproj.filters")
filtered = {n.get("Include") for n in filters.findall(".//m:ClCompile[@Include]", NS)}
check(set(compiled) == filtered, "debug filters match compiled sources")
for filename in ("ModuleDebug.cpp", "ModuleDebugMain.cpp", "ModuleDebug.hpp", "ModuleDebugArtifacts.hpp"):
    text = (ROOT / "src/debug" / filename).read_text(encoding="utf-8")
    clean = re.sub(strip, " ", text, flags=re.S)
    stack = []; balanced = True
    for ch in clean:
        if ch in "([{": stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[ch]: balanced = False; break
    check(balanced and not stack, filename + ": delimiters balanced")
runner = (ROOT / "src/debug/ModuleDebug.cpp").read_text(encoding="utf-8")
check('"status.json"' in runner and 'INCOMPLETE / RUNNING' in runner, "incomplete runs cannot report all-clear")
check('synthetic controlled checks;' in runner, "synthetic evidence is labelled")
check('root/"latest.txt"' in runner, "latest report pointer persisted")
check('catch(const std::exception& error)' in runner and 'return failures?1:0;' in runner, "test failures have explicit status and exit code")
check("google::AddLogSink(this)" in runner and "google::RemoveLogSink(this)" in runner, "module log sink is scoped")
backend = (ROOT / "src/backend/tests/ManagedWindowRegression.cpp").read_text(encoding="utf-8")
check('"_jacobian.csv"' in backend and '"conditional_cost_delta_"' in backend, "Jacobian and Schur numeric evidence emitted")
check('recordError("schur_information_error"' in backend, "Schur result checked against independent information")
artifacts = (ROOT / "src/debug/ModuleDebugArtifacts.hpp").read_text(encoding="utf-8")
check("absoluteTolerance+relativeTolerance*referenceNorm" in artifacts, "mixed derivative tolerance uses actual reference norm")
check('"derivative_checks.csv"' in artifacts, "derivative tolerance policy exported")
pose = (ROOT / "src/frontend/tests/FrontendPoseRegression.cpp").read_text(encoding="utf-8")
for name in ("absoluteUnseededRecovery", "absoluteNoisyOutlierRecovery", "absoluteWrongPredictionRecovery",
             "relativeUnseededRecovery", "relativeNoisyOutlierRecovery", "relativeWrongPredictionRecovery"):
    check(("tracking", name) in names, "independent recovery registered: " + name)
check("?&wrong:nullptr" in pose and "result.generatedModels>0" in pose, "new recovery controls do not pass truth as initial model")
check('"pose_recovery.csv"' in pose and '"_correspondences.csv"' in pose, "pose truth labels and numeric recovery evidence exported")
matching = (ROOT / "src/frontend/tests/FrontendOrbRegression.cpp").read_text(encoding="utf-8")
check("require(!actual.empty() && statistics.candidatePairs" not in matching, "empty matching semantics separated from candidate visit bound")
check('"matching_policy_combinations.csv"' in matching and "if(single) require(!actual.empty()" in matching, "all policies exported with nonempty positive control")
association = (ROOT / "src/frontend/tests/FrontendBackendRegression.cpp").read_text(encoding="utf-8")
check("require(broad.broadFallback && broad.seedJobs" not in association and "expanded==expected" in association, "fallback verifies actual task coverage and point identities")
audit = (ROOT / "src/backend/LandmarkReanchoring.cpp").read_text(encoding="utf-8")
check(audit.index("if(!audit.equivalent)") < audit.index("item.second=replacement;"), "production rejects non-equivalent candidate before mutation")
check("true,true,oldSystem" in audit and "true,true,newSystem" in audit, "reanchor audit includes historical prior and robustification")
for name in ("reanchorInformationAudit", "reanchorFixedBearingChangeIsRejected", "reanchorFullCoordinateTransportReference"):
    check(("marginalization", name) in names, "reanchor validation registered: " + name)
check("validationPolicy" in runner, "report identifies strengthened validation policy")
check(("tracking", "stationaryKeyframesPreserveMatureAnchors") in names, "textured stationary window retention regression registered")
check(("backend", "realWindowResidualStatisticsContract") in names, "residual phase and whitening/robust-cost contract registered")
window = (ROOT / "src/backend/SlidingWindowManager.cpp").read_text(encoding="utf-8")
check(window.index("detail.survivingLive") < window.index("auto result = marginalizer_.compute(candidate"), "retirement evidence captured before Schur consumes observations")
kernel = (ROOT / "src/backend/WindowVisualFactor.cpp").read_text(encoding="utf-8")
check("pc=Rcb_.transpose()*(local-tcb_)" in kernel and "if(!analytic_) return evaluateNumerical" in kernel,
      "optimized rig stereo kernel retains independent numerical reference")
check("projected-(projected*n)*n.transpose()" in kernel, "visual derivative uses projected rank-one normalization")
check("resetPriorUnchanged=true" in window and "chart preparation changed a prior-coupled coordinate" in window,
      "chart reset protects incoming frozen prior coordinates")
check("if(a.frameId!=b.frameId) return a.frameId>b.frameId" in window, "reset prefers a recent surviving keyframe")
native = (ROOT / "src/pipeline/tests/NativeSlamRegression.cpp").read_text(encoding="utf-8")
check("dynamic_runtime_audit_scaled_error" in native, "runtime IMU audit has a dynamic numerical regression")
for module, name in (("jacobian", "visualPoseCacheMatchesUncachedTrials"),
                     ("jacobian", "fractionalIncrementalImuMatchesFullReplay"),
                     ("marginalization", "oldestSchurKeepsIdentityWithoutReusingEvidence"),
                     ("marginalization", "longImuIntervalsAreStorageOnly"),
                     ("backend", "nonKeyframeRetainsSeedsWithoutRho")):
    check((module, name) in names, "v9 reference regression registered: " + name)
for name in ("nonKeyframeRetainsSeedsWithoutRho", "seedAdmissionIsBoundedAndExpiresAtomically", "seedReferenceUpdateCannotRewriteOptimizedGeometry"):
    check(("backend", name) in names, "v15 tracking seed lifecycle regression registered: " + name)
for module,name in (("matching","existingIdentityWinsBeforeNewTexture"),
                    ("tracking","seedReplenishmentPreservesExistingCoverage"),
                    ("jacobian","rotationAnalyticAgreementAndRecovery")):
    check((module,name) in names,"v16 continuity/performance regression registered: "+name)
check("retainExistingTracks(plan)" not in (ROOT / "src/frontend/Frontend.cpp").read_text(encoding="utf-8"),
      "non-keyframe admission no longer deletes all new track identities")
check('"DENSE_SCHUR"' in backend and "denseCoupled" in backend,
      "dense Schur regression covers solution agreement and coupled-prior fallback")
check("deliberately unevaluable long cost" in native and "prior.consumed.empty()" in native,
      "long interval regression covers both optimizer and marginalizer exclusion")
check("historical_visual_information_absorbed" in window and "candidateLandmarks.absorbedKeys_.insert(view)" in window,
      "Schur identity reuse consumes implicit anchor rays too")
check("candidateLandmarks.landmarks_" in window and "incident_range_schur_retired" in window,
      "strict retirement also removes incident ranges with retained anchors")
pipeline = (ROOT / "src/pipeline/ThreadedSlam.cpp").read_text(encoding="utf-8")
check("options.maxSolverTimeSeconds = p.estimator.enforce_realtime ? p.estimator.realtime_time_limit" in pipeline
      and "pendingBudget_.timeSeconds = options_.maxSolverTimeSeconds;" in pipeline,
      "solver budget no longer depends on offline queue blocking mode")
check("realtime_time_limit -" not in pipeline and "realtime_time_limit-" not in pipeline,
      "frontend staging does not debit the Ceres-only allowance")
optimizer = (ROOT / "src/backend/BackendOptimizer.cpp").read_text(encoding="utf-8")
check("timeLimit - out.ceresMs*1e-3" in optimizer and "timer.elapsed=[&]()" in optimizer
      and "solveStarted=std::chrono::steady_clock::now();\n                        ::ceres::Solve" in optimizer,
      "refits share only Ceres calls and start timing immediately before Solve")
check(("marginalization", "budgetLimitedSolvePreservesCleanPrior") in names,
      "budget-limited accepted state has a clean-prior regression")
check("out.accepted=true; out.termination=termination" in optimizer
      and "out.postSolveResiduals=survivingResiduals" in optimizer,
      "a solve limit retains only a quality-checked state with cleaned residual statistics")
check("SolveTermination::Failure" in backend and "rejectQuality" in backend,
      "budget-limited quality failure still has rollback coverage")
check(("backend", "constraintGradientBalanceIsRobustAndReadOnly") in names,
      "v10 robust current-tangent gradient regression registered")
check("next.ignoredAbsorbedIncidentObservations>0" in backend and "retainedCost" in backend,
      "OLD regression repeats eviction after SAME identity rematures on fresh stereo")
check("formation_constant_offset" in backend and "formation_eliminated_reduction" in backend,
      "Schur formation diagnostics compared with an independent scalar quadratic")
check("deferredEvictionSupport==1" in association,
      "promotion regression covers retained anchor and fresh outgoing support")
check("accepted_trajectory.png" in native and "accepted_feature_tracks.png" in native,
      "accepted trajectory and track rendering have module artifacts")
for module, name in (("jacobian", "absoluteAnalyticLinearizationAgreement"),
                     ("tracking", "absoluteAnalyticRecoveryAndBudgetGuard"),
                     ("extraction", "adaptiveGeometryBatchesPreserveSupport")):
    check((module, name) in names, "v11 CPU optimization regression registered: " + name)
check('cached_partial_' in backend and 'cache.pairCount()==1' in backend,
      "pair cache regression covers sharing, trials and partial Jacobian requests")
solution = (ROOT / "CGMLI-SLAM/CGMLI-SLAM.sln").read_text(encoding="utf-8-sig")
check('"CGMLI-ModuleDebug.vcxproj"' in solution, "debug project available in solution")
for module, name in (("backend", "cachedVisualGateMatchesReference"), ("backend", "healthyPromotionBudgetPreservesSeeds"),
                     ("extraction", "batchedDescriptorCoordinatesMatchScalar")):
    check((module, name) in names, "v12 CPU path regression registered: " + name)
failures = [c for c in checks if not c["passed"]]
report = {"kind": "static module-debug wiring inspection; C++ not compiled or executed", "cases": len(names), "modules": sorted(modules), "passed": len(checks)-len(failures), "total": len(checks), "checks": checks}
(ROOT / "src/debug/tools/module_debug_static_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Module debug static checks: {report['passed']}/{report['total']}; {len(names)} cases in {len(modules)} modules. No C++ build/execution.")
for failure in failures: print("FAIL:", failure["name"])
raise SystemExit(bool(failures))
