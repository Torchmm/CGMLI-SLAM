// Standalone regression source. Not part of the application target and NOT executed
// in the static-only implementation task. Link backend + project camera/time sources.
#include <debug/ModuleDebug.hpp>
#include <debug/ModuleDebugArtifacts.hpp>
#include <backend/ViSlamBackend.hpp>
#include <backend/BackendFactors.hpp>
#include <backend/BackendOptimizer.hpp>
#include <backend/WindowVisualFactor.hpp>
#include <backend/BackendProblem.hpp>
#include <cameras/PinholeCamera.hpp>
#include <cameras/EucmCamera.hpp>
#include <visual/MultiFrame.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <atomic>
#include <chrono>
#include <thread>

using namespace cgmli;
using namespace cgmli::backend;
namespace {
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
struct Fixture {
    cameras::NCameraSystem rig;
    std::vector<Eigen::Vector3d> points{Eigen::Vector3d(0.3,0.1,4),Eigen::Vector3d(-0.4,0.2,5),
        Eigen::Vector3d(0.2,-0.3,3),Eigen::Vector3d(1,0,-0.2)};
    Fixture() {
        rig.addCamera(std::make_shared<kinematics::Transformation>(),
            std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(640,480,300,300,320,240,cameras::NoDistortion()),
            cameras::NCameraSystem::NoDistortion,false);
        rig.addCamera(std::make_shared<kinematics::Transformation>(Eigen::Vector3d(0.2,0,0),Eigen::Quaterniond::Identity()),
            std::make_shared<cameras::EucmCamera>(640,480,120,120,320,240,0.75,1.0),
            cameras::NCameraSystem::NoDistortion,false);
    }
    StateEstimate prediction(uint64_t id) const {
        StateEstimate s; s.id=StateId(id); s.timestamp=Time(static_cast<uint32_t>(id),0);
        s.T_WB=kinematics::Transformation(Eigen::Vector3d(0.1*(id-1),0,0),Eigen::Quaterniond::Identity());
        s.visual=SensorHealth::Valid; s.lidar=SensorHealth::Valid; s.keyframe=true; return s;
    }
    std::shared_ptr<MultiFrame> frame(uint64_t id) const {
        const auto state=prediction(id);
        auto result=std::make_shared<MultiFrame>(rig,state.timestamp,id);
        for(size_t c=0;c<rig.numCameras();++c) {
            const auto cameraPose=state.T_WB*(*rig.T_SC(c));
            std::vector<cv::KeyPoint> keypoints;
            for(const auto& world:points) {
                const Eigen::Vector3d pc=cameraPose.C().transpose()*(world-cameraPose.r());
                Eigen::Vector2d pixel;
                if(rig.cameraGeometry(c)->project(pc,&pixel)!=cameras::ProjectionStatus::Successful) break;
                keypoints.emplace_back(static_cast<float>(pixel.x()),static_cast<float>(pixel.y()),31.0f);
            }
            result->resetKeypoints(c,keypoints);
            result->resetDescriptors(c,cv::Mat::zeros(static_cast<int>(keypoints.size()),32,CV_8UC1));
            result->setImage(c,cv::Mat::ones(480,640,CV_8UC1));
        }
        return result;
    }
    double rho(uint64_t frame,size_t camera,size_t point=0) const {
        const auto pose=prediction(frame).T_WB*(*rig.T_SC(camera)); return 1.0/(points[point]-pose.r()).norm();
    }
};
void initialize(FixedLagBackend& backend,const Fixture& f) {
    std::string error;
    require(backend.initializeCalibration(makeRigCalibration(f.rig,kinematics::Transformation()).parameterBlocks(),error),"rig initialization");
}
void append(FixedLagBackend& backend,const Fixture& f,uint64_t id) {
    std::string error; auto frame=f.frame(id);
    require(backend.append(f.prediction(id),*frame,ImuMeasurementDeque(),error),"managed append");
}
void poseConstraints(FixedLagBackend& backend,StateId id) {
    const auto& graph=backend.window().graph();
    require(backend.addFactor(makeParameterPrior(graph.blocks.at(BlockKey::pose(id)),10.0*Eigen::Matrix<double,6,6>::Identity())).accepted(),"pose test anchor");
    require(backend.addFactor(makeParameterPrior(graph.blocks.at(BlockKey::speedBias(id)),Eigen::Matrix<double,9,9>::Identity())).accepted(),"velocity/bias test anchor");
}
void ownershipAndIndices() {
    Fixture f; BackendOptions options; options.windowSize=3;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    auto input=f.frame(1); require(backend.append(f.prediction(1),*input,{},error),"frame attach");
    input->resetDescriptors(0,cv::Mat::ones(3,32,CV_8UC1));
    require(backend.window().frames().measurements(StateId(1))->descriptors(0).at<unsigned char>(0,0)==0,"descriptor buffer owned, not aliased");
    require(backend.window().frames().measurements(StateId(1))->image(0).empty(),"raw images not retained by default");
    require(backend.addLandmark(LandmarkId(1),{1,0,0},std::nullopt,error),"seed creation");
    require(!backend.window().graph().blocks.count({BlockKind::Landmark,1}),"seed has no fake optimized depth");
    require(backend.addObservation(LandmarkId(1),{1,1,0},error),"cross-camera seed observation");
    require(!backend.addObservation(LandmarkId(1),{1,1,1},error),"one landmark per camera/epoch");
    require(backend.initializeLandmark(LandmarkId(1),f.rho(1,0),error),"promote seed atomically");
    const auto observation=backend.window().landmarks().find(KeypointIdentifier{1,1,0});
    require(observation && observation->use==ObservationUse::LiveFactor,"initialized observation owns a factor binding");
    const auto& factor=backend.window().graph().factors.at(observation->factorId);
    const auto factorId=factor.id;
    require(std::count(factor.keys.begin(),factor.keys.end(),BlockKey::pose(StateId(1)))==1,"same-epoch cameras coalesce pose block");
    MapPoint2 point; require(backend.window().getLandmark(LandmarkId(1),point),"project MapPoint2 adapter");
    require(point.observations.size()==2 && (point.point.head<3>()-f.points[0]).norm()<1e-5,"world point derived from authoritative rho/poses");
    auto view=backend.window().exportFrame(StateId(1));
    require(view && view->numKeypoints(0)==3 && view->landmarkId(1,0)==1,"feature reconstruction and association export");
    auto target=f.frame(1); const auto* descriptorBuffer=target->descriptors(0).data;
    target->setLandmarkId(0,2,999); // Stale caller associations must be cleared.
    backend.window().synchronizeFrame(StateId(1),*target);
    require(target->landmarkId(1,0)==1 && target->landmarkId(0,2)==0
        && target->descriptors(0).data==descriptorBuffer,"association refresh preserves feature storage and clears stale IDs");
    require(backend.removeObservation({1,1,0},error),"remove observation");
    require(!backend.window().landmarks().find(KeypointIdentifier{1,1,0}) && !backend.window().graph().factors.count(factorId),"factor and both indices removed together");
    require(!backend.removeObservation({1,0,0},error),"geometric anchor cannot be erased independently");
    require(backend.window().valid(error),"managed graph invariants");
    auto raw=ParameterBlock::euclidean({BlockKind::Landmark,2},Eigen::VectorXd::Ones(1),StateId(1));
    require(!backend.addParameter(raw,error),"raw landmark insertion cannot bypass geometric identity");
    VisualUpdateBatch batch;
    batch.landmarks.push_back({LandmarkId(2),{1,0,1},std::nullopt,{}});
    batch.observations.push_back({LandmarkId(2),{1,0,999},{}});
    require(!backend.applyVisualUpdates(batch,error) && !backend.window().landmarks().find(LandmarkId(2)),"invalid batch rolls back earlier track insertion");
    batch.observations[0].key={1,1,1};
    require(backend.applyVisualUpdates(batch,error) && backend.window().landmarks().ofLandmark(LandmarkId(2)).size()==2,"one transaction ingests a full association batch");
}
void factorGeometryAndJacobians() {
    Fixture f; FixedLagBackend backend; initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"initialize geometric point");
    ObservationOptions observationOptions; observationOptions.bearingVelocity=Eigen::Vector3d::Zero();
    require(backend.addObservation(LandmarkId(1),{2,1,0},error,observationOptions),"mixed-model observation");
    const auto observation=backend.window().landmarks().find(KeypointIdentifier{2,1,0});
    const auto& factor=backend.window().graph().factors.at(observation->factorId);
    std::vector<const double*> pointers;
    std::vector<std::vector<double>> storage;
    std::vector<double*> jacobians;
    for(auto key:factor.keys) {
        const auto& block=backend.window().graph().blocks.at(key); pointers.push_back(block.value.data());
        storage.emplace_back(2*block.value.size());
    }
    for(auto& values:storage) jacobians.push_back(values.data());
    Eigen::Vector2d residual; require(factor.cost->Evaluate(pointers.data(),residual.data(),jacobians.data()),"S2 factor Evaluate");
    require(residual.norm()<1e-3,"camera-independent whitened residual");
    const LeftSE3Manifold manifold;
    for(size_t k=0;k<factor.keys.size();++k) {
        const auto& block=backend.window().graph().blocks.at(factor.keys[k]);
        const int ambient=static_cast<int>(block.value.size()),tangent=block.tangentSize();
        Eigen::Map<Eigen::Matrix<double,2,Eigen::Dynamic,Eigen::RowMajor>> J(jacobians[k],2,ambient);
        Eigen::MatrixXd analytic=J;
        if(block.manifold) {
            Eigen::Matrix<double,7,6,Eigen::RowMajor> P;
            require(manifold.PlusJacobian(block.value.data(),P.data()),"left pose lift"); analytic=J*P;
        }
        for(int column=0;column<tangent;++column) {
            constexpr double h=2e-7;
            Eigen::VectorXd plus=block.value,minus=block.value,step=Eigen::VectorXd::Zero(tangent);
            if(block.manifold) {
                step[column]=h; manifold.Plus(block.value.data(),step.data(),plus.data());
                step[column]=-h; manifold.Plus(block.value.data(),step.data(),minus.data());
            } else { plus[column]+=h; minus[column]-=h; }
            Eigen::Vector2d rp,rm;
            pointers[k]=plus.data(); require(factor.cost->Evaluate(pointers.data(),rp.data(),nullptr),"FD plus");
            pointers[k]=minus.data(); require(factor.cost->Evaluate(pointers.data(),rm.data(),nullptr),"FD minus");
            pointers[k]=block.value.data();
            const Eigen::Vector2d numeric=(rp-rm)/(2*h);
            debug::recordDerivative("visual_tangent_block_"+std::to_string(k)+"_column_"+std::to_string(column),
                (numeric-analytic.col(column)).norm(),numeric.norm());
        }
    }
    require(f.frame(1)->numKeypoints(1)==4,"rear-hemisphere EUCM fixture");
    require(backend.addLandmark(LandmarkId(2),{1,1,3},f.rho(1,1,3),error),"negative-z anchor");
    require(backend.window().landmarks().find(LandmarkId(2))->anchorBearing.z()<0,"full-sphere bearing retained");
    require(backend.addObservation(LandmarkId(2),{2,1,3},error),"negative-z observation");
}
void analyticSphereDerivativeAgreement() {
    Fixture f; FixedLagBackend backend; initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"analytic normal anchor");
    require(backend.addObservation(LandmarkId(1),{1,1,0},error),"analytic stereo view");
    require(backend.addObservation(LandmarkId(1),{2,1,0},error),"analytic temporal view");
    require(backend.addLandmark(LandmarkId(2),{1,1,3},f.rho(1,1,3),error),"analytic rear anchor");
    require(backend.addObservation(LandmarkId(2),{2,1,3},error),"analytic rear view");
    for(const auto key:std::vector<KeypointIdentifier>{{1,1,0},{2,1,0},{2,1,3}}) for(int scenario=0;scenario<4;++scenario) {
        auto graph=backend.window().graph(); auto observation=*backend.window().landmarks().find(key);
        const auto landmark=*backend.window().landmarks().find(observation.landmark);
        auto anchor=*backend.window().landmarks().find(landmark.anchor);
        if(scenario) {
            const auto velocity=[](const Eigen::Vector3d& b) { return Eigen::Vector3d(0.15*(Eigen::Matrix3d::Identity()-b*b.transpose())*Eigen::Vector3d(1,2,-1)); };
            anchor.bearingVelocity=velocity(anchor.bearing); observation.bearingVelocity=velocity(observation.bearing);
            anchor.referenceTimeOffset=0.004; observation.referenceTimeOffset=-0.003;
            graph.blocks.at({BlockKind::CameraTimeOffset,0}).value[0]=scenario==1?0.001:0.004;
            graph.blocks.at({BlockKind::CameraExtrinsic,0}).value=ParameterBlock::pose({BlockKind::CameraExtrinsic,0},
                kinematics::Transformation(Eigen::Vector3d(0.04,-0.02,0.01),expSO3(Eigen::Vector3d(0.09,-0.06,0.04)))).value;
            if(scenario>=2) {
                auto& state=graph.blocks.at(BlockKey::pose(StateId(2)));
                state.value=ParameterBlock::pose(state.key,kinematics::Transformation(Eigen::Vector3d(0.15,0.02,-0.03),
                    expSO3(Eigen::Vector3d(0.07,-0.03,0.1))),state.owner).value;
                graph.blocks.at({BlockKind::Landmark,landmark.id.value()}).value[0]*=scenario==2?0.6:1.8;
            }
        }
        BackendOptions options; FactorRecord analytic,numeric;
        require(makeWindowVisualFactor(landmark,anchor,observation,backend.window().frames(),graph,options,analytic,error),"analytic factor");
        options.analyticVisualJacobians=false;
        require(makeWindowVisualFactor(landmark,anchor,observation,backend.window().frames(),graph,options,numeric,error),"numeric factor");
        std::vector<const double*> parameters; std::vector<std::vector<double>> a,n;
        for(auto block:analytic.keys) {
            parameters.push_back(graph.blocks.at(block).value.data());
            a.emplace_back(size_t(2*graph.blocks.at(block).value.size())); n.push_back(a.back());
        }
        std::vector<double*> ja,jn;
        for(size_t i=0;i<a.size();++i) { ja.push_back(a[i].data()); jn.push_back(n[i].data()); }
        Eigen::Vector2d ra,rn;
        require(analytic.cost->Evaluate(parameters.data(),ra.data(),ja.data()) && numeric.cost->Evaluate(parameters.data(),rn.data(),jn.data()),"analytic/reference evaluation");
        const std::string prefix="frame_"+std::to_string(key.frameId)+"_landmark_"+std::to_string(landmark.id.value())+"_scenario_"+std::to_string(scenario);
        debug::recordError(prefix+"_residual_error",(ra-rn).norm(),1e-8*(1+rn.norm()));
        auto csv=debug::artifactFile(prefix+"_jacobian.csv");
        if(csv) csv<<"block,block_kind,variable_id,row,column,analytic,numeric,absolute_error\n";
        require((ra-rn).norm()<1e-8*(1+rn.norm()),"Analytic path changed the whitened spherical residual");
        for(size_t i=0;i<a.size();++i) {
            Eigen::Map<const Eigen::VectorXd> actual(a[i].data(),a[i].size()),expected(n[i].data(),n[i].size());
            if(csv) for(Eigen::Index component=0;component<actual.size();++component)
                csv<<i<<','<<int(analytic.keys[i].kind)<<','<<analytic.keys[i].id<<','<<component/(actual.size()/2)<<','<<component%(actual.size()/2)
                    <<','<<actual[component]<<','<<expected[component]<<','<<std::abs(actual[component]-expected[component])<<'\n';
            const auto kind=analytic.keys[i].kind;
            const double tolerance=(kind==BlockKind::CameraTimeOffset || kind==BlockKind::CameraExtrinsic)?1e-5:1e-6;
            debug::recordDerivative(prefix+"_block_"+std::to_string(i),(actual-expected).norm(),expected.norm(),tolerance);
            auto partial=ja;
            for(size_t k=0;k<partial.size();++k) if(k!=i) partial[k]=nullptr;
            require(analytic.cost->Evaluate(parameters.data(),ra.data(),partial.data()),"Partial Jacobian requests must be supported");
            Eigen::Vector2d residualOnly;
            require(analytic.cost->Evaluate(parameters.data(),residualOnly.data(),nullptr),"residual-only fast path");
            debug::recordError(prefix+"_residual_only_"+std::to_string(i),(residualOnly-ra).norm(),1e-11);
        }
        if(key.frameId==landmark.anchor.frameId) require(Eigen::Map<Eigen::VectorXd>(a[0].data(),a[0].size()).norm()==0
            && Eigen::Map<Eigen::VectorXd>(a[1].data(),a[1].size()).norm()==0,"Same-epoch pose/shared extrinsic must cancel exactly");
        if(scenario==3) {
            graph.blocks.at({BlockKind::Landmark,landmark.id.value()}).value[0]=0;
            require(!analytic.cost->Evaluate(parameters.data(),ra.data(),nullptr),"Invalid inverse depth must still reject");
        }
    }
}
void marginalizationQuadraticAgreement() {
    class LinearCost final : public ::ceres::SizedCostFunction<2,1,1> {
        bool Evaluate(double const* const* x,double* r,double** J) const override {
            r[0]=2*x[0][0]+x[1][0]-3; r[1]=x[0][0]+3*x[1][0]+1;
            if(J && J[0]) { J[0][0]=2; J[0][1]=1; }
            if(J && J[1]) { J[1][0]=1; J[1][1]=3; }
            return true;
        }
    };
    BackendGraph graph; const BlockKey m{BlockKind::Auxiliary,1},r{BlockKind::Auxiliary,2};
    graph.blocks.emplace(m,ParameterBlock::euclidean(m,Eigen::VectorXd::Constant(1,0.3)));
    graph.blocks.emplace(r,ParameterBlock::euclidean(r,Eigen::VectorXd::Constant(1,-0.4)));
    FactorRecord factor; factor.id=1; factor.keys={m,r}; factor.lifecycle=FactorLifecycle::Active;
    factor.cost=std::make_shared<LinearCost>(); graph.factors.emplace(1,factor);
    const auto result=Marginalizer(BackendOptions()).compute(graph,{m});
    require(result.success && result.prior && result.prior->keys()==std::vector<BlockKey>{r},"Linear Schur prior must retain only the separator");
    const auto& A=result.prior->squareRootInformation(); const auto& b=result.prior->residualAtReference();
    const Eigen::MatrixXd information=A.transpose()*A; const Eigen::VectorXd gradient=A.transpose()*b;
    debug::recordError("schur_information_error",std::abs(information(0,0)-5),1e-10);
    debug::recordError("schur_gradient_error",std::abs(gradient[0]-3),1e-10);
    require(result.formation.evaluated && result.formation.otherFactors==1,"formation audit counts each consumed cost once");
    debug::recordError("formation_input_quadratic",std::abs(result.formation.linearizedSquaredNorm-7.85),1e-10);
    debug::recordError("formation_robust_cost",std::abs(result.formation.robustCost-3.925),1e-10);
    debug::recordError("formation_eliminated_reduction",std::abs(result.formation.eliminatedReduction-3.025),1e-10);
    debug::recordError("formation_constant_offset",std::abs(result.formation.quadraticConstantOffset),1e-10);
    auto formationCsv=debug::artifactFile("prior_formation.csv");
    if(formationCsv) formationCsv<<"robust_cost,linearized_squared_norm,eliminated_reduction,new_reference_norm,quadratic_constant_offset\n"
        <<result.formation.robustCost<<','<<result.formation.linearizedSquaredNorm<<','<<result.formation.eliminatedReduction<<','
        <<result.formation.newReferenceNorm<<','<<result.formation.quadraticConstantOffset<<'\n';
    auto csv=debug::artifactFile("quadratic_equivalence.csv");
    if(csv) csv<<"delta_retained,conditional_full_cost,prior_cost,delta_cost_error\n";
    const double base=0.9;
    for(double delta:{-1.0,-0.1,0.0,0.1,1.0}) {
        const double eliminated=1.4-delta,retained=-0.4+delta;
        const double* full[]{&eliminated,&retained}; Eigen::Vector2d residual;
        require(factor.cost->Evaluate(full,residual.data(),nullptr),"Full linear residual");
        Eigen::VectorXd prior(result.prior->num_residuals()); const double* input[]{&retained};
        require(result.prior->Evaluate(input,prior.data(),nullptr),"Reduced linear residual");
        const double error=std::abs((0.5*residual.squaredNorm()-base)-(0.5*prior.squaredNorm()-0.5*b.squaredNorm()));
        if(csv) csv<<delta<<','<<0.5*residual.squaredNorm()<<','<<0.5*prior.squaredNorm()<<','<<error<<'\n';
        debug::recordError("conditional_cost_delta_"+std::to_string(delta),error,1e-10);
    }
    require(graph.factors.size()==1 && graph.blocks.size()==2 && !graph.prior,"Computing a prior must not mutate input topology");
}
void parallelNormalEquationsPreservePrior() {
    BackendGraph graph;
    const BlockKey victim{BlockKind::Auxiliary,1}, retained{BlockKind::Auxiliary,2};
    graph.blocks.emplace(victim,ParameterBlock::euclidean(victim,Eigen::VectorXd::Constant(4,0.03)));
    graph.blocks.emplace(retained,ParameterBlock::euclidean(retained,Eigen::VectorXd::Constant(4,-0.08)));
    Eigen::MatrixXd J(384,8);
    for (int row=0; row<J.rows(); ++row)
        for (int column=0; column<J.cols(); ++column) J(row,column)=std::sin(0.011*(row+1)*(column+1));
    Eigen::VectorXd r=Eigen::VectorXd::LinSpaced(J.rows(),-0.15,0.4);
    FactorRecord factor; factor.id=1; factor.keys={victim,retained}; factor.lifecycle=FactorLifecycle::Active;
    factor.cost=std::make_shared<MarginalizationPrior>(
        std::vector<ParameterBlock>{graph.blocks.at(victim),graph.blocks.at(retained)},J,r);
    factor.loss=std::make_shared<::ceres::HuberLoss>(1.0);
    graph.factors.emplace(factor.id,std::move(factor));
    BackendOptions options; options.numThreads=4; options.parallelNormalEquations=false;
    Marginalizer reference(options);
    options.parallelNormalEquations=true;
    Marginalizer parallel(options);
    LinearizedSystem a,b; std::string error;
    require(reference.linearize(graph,{1},false,true,a,error) && parallel.linearize(graph,{1},false,true,b,error)
        && a.timing.normalThreads==1 && b.timing.normalThreads>1,"exercise both robust normal-equation paths");
    debug::recordError("parallel_normal_H",(a.H-b.H).norm(),1e-9);
    debug::recordError("parallel_normal_gradient",(a.b-b.b).norm(),1e-10);
    const auto x=reference.compute(graph,{victim}), y=parallel.compute(graph,{victim});
    require(x.success && y.success && x.prior && y.prior && x.priorRank==y.priorRank
        && y.linearization.normalThreads>1,"parallel reduction preserves prior rank");
    const auto& Ax=x.prior->squareRootInformation(); const auto& Ay=y.prior->squareRootInformation();
    debug::recordError("parallel_prior_H",(Ax.transpose()*Ax-Ay.transpose()*Ay).norm(),1e-9);
    debug::recordError("parallel_prior_gradient",(Ax.transpose()*x.prior->residualAtReference()
        -Ay.transpose()*y.prior->residualAtReference()).norm(),1e-10);
}
void untouchedPriorAndDiagonalKernel() {
    BackendOptions options; Marginalizer marginalizer(options); BackendGraph graph;
    const BlockKey victim{BlockKind::Auxiliary,90},a{BlockKind::Auxiliary,91},b{BlockKind::Auxiliary,92};
    for(auto key:{victim,a,b}) graph.blocks.emplace(key,ParameterBlock::euclidean(key,Eigen::VectorXd::Constant(1,0.3)));
    Eigen::Matrix2d A; A<<2,0.4,-0.3,1.5;
    graph.prior=std::make_shared<MarginalizationPrior>(std::vector<ParameterBlock>{graph.blocks.at(a),graph.blocks.at(b)},A,Eigen::Vector2d(0.2,-0.1));
    const auto frozen=graph.prior; graph.blocks.at(a).value[0]+=0.2;
    const auto result=marginalizer.compute(graph,{victim});
    require(result.success && result.priorReused && result.prior==frozen && result.consumed.empty(),
        "unrelated eviction must preserve the exact prior object and frozen references");
    require(result.prior->references()[0][0]==0.3,"fast path must not recenter FEJ reference");
    auto broken=graph; broken.blocks.erase(b);
    require(!marginalizer.compute(broken,{victim}).success,"prior reuse must still validate every frozen dependency");
    Eigen::Matrix4d H=Eigen::Matrix4d::Zero(); H.diagonal()<<2,0,1e-14,5;
    const Eigen::Matrix<double,4,2> rhs=Eigen::Matrix<double,4,2>::Ones(); Eigen::MatrixXd actual;
    std::string error; require(marginalizer.psdSolve(H,rhs,actual,error),"diagonal PSD kernel");
    Eigen::Matrix<double,4,2> expected=Eigen::Matrix<double,4,2>::Zero(); expected.row(0).setConstant(0.5); expected.row(3).setConstant(0.2);
    debug::recordError("diagonal_rank_aware_solve",(actual-expected).norm(),1e-12);
    H(2,2)=-1; require(!marginalizer.psdSolve(H,rhs,actual,error),"negative diagonal information must fail");
}
void redundantFrameKeepsTracksSparse() {
    Fixture f; BackendOptions options; options.windowSize=2;
    options.preserveOldestVisualInformation=false;
    options.resetOldestLandmarks=false; // Explicitly exercise conservative OLD retirement below.
    SlidingWindowManager window(options); std::string error;
    for(auto block:makeRigCalibration(f.rig,kinematics::Transformation()).parameterBlocks())
        require(window.addParameter(std::move(block),error),"redundancy rig calibration");
    for(uint64_t i=1;i<=3;++i) require(window.append(f.prediction(i),*f.frame(i),error),"redundancy append");
    require(window.addFactor(makeParameterPrior(window.graph().blocks.at(BlockKey::pose(StateId(1))),
        10.0*Eigen::Matrix<double,6,6>::Identity())).accepted(),"mixed-policy pose gauge");
    require(window.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"retained keyframe anchor");
    require(window.addObservation(LandmarkId(1),{2,0,0},error) && window.addObservation(LandmarkId(1),{3,0,0},error),"retained point views");
    require(window.addLandmark(LandmarkId(2),{2,0,1},f.rho(2,0,1),error),"redundant mature anchor");
    require(window.addObservation(LandmarkId(2),{3,0,1},error) && window.addObservation(LandmarkId(2),{3,1,1},error),"two surviving mature views");
    require(window.addLandmark(LandmarkId(3),{2,0,2},std::nullopt,error)
        && window.addObservation(LandmarkId(3),{3,0,2},error),"surviving seed");
    auto trace=debug::artifactFile("redundant_window.csv");
    if(trace) trace<<"frame,reset_tracks,moved_seeds,dropped_visual,prior_landmarks,prior_reused\n";
    for(uint64_t id=3;id<=8;++id) {
        if(id>3) {
            require(window.append(f.prediction(id),*f.frame(id),error),"next redundancy append");
            require(window.addObservation(LandmarkId(1),{id,0,0},error),"continue stable track");
            require(window.addObservation(LandmarkId(2),{id,0,1},error) && window.addObservation(LandmarkId(2),{id,1,1},error),"continue mature track");
            require(window.addObservation(LandmarkId(3),{id,0,2},error),"continue seed identity");
        }
        const auto report=window.marginalize(MarginalizationPolicy::MARGIN_SECOND_NEW);
        require(report.success && report.priorReused && !window.graph().prior && report.priorLandmarks==0,
            "redundant image measurements must not create a landmark prior");
        require(window.landmarks().find(LandmarkId(1)) && window.landmarks().find(LandmarkId(2))
            && window.landmarks().find(LandmarkId(3)),"redundancy preserves stable, reset and seed IDs");
        require(report.resetLandmarks==std::vector<LandmarkId>{LandmarkId(2)} && report.movedSeeds==1,
            "mature reset and seed reference move have distinct receipts");
        require(window.landmarks().find(LandmarkId(2))->anchor.frameId==id
            && window.landmarks().find(LandmarkId(3))->anchor.frameId==id,"outgoing anchors must not dangle");
        Eigen::Vector3d world; require(window.landmarkPoint(LandmarkId(2),world),"reset mature geometry");
        debug::recordError("reset_world_point_"+std::to_string(id),(world-f.points[1]).norm(),1e-4);
        require(window.valid(error),"redundant graph and observation indexes remain consistent");
        if(trace) trace<<id<<','<<report.resetLandmarks.size()<<','<<report.movedSeeds<<','<<report.droppedVisualFactors<<','<<report.priorLandmarks<<','<<report.priorReused<<'\n';
    }
    require(window.append(f.prediction(9),*f.frame(9),error),"oldest-policy append");
    require(window.addObservation(LandmarkId(1),{9,0,0},error)
        && window.addObservation(LandmarkId(2),{9,0,1},error)
        && window.addObservation(LandmarkId(2),{9,1,1},error)
        && window.addObservation(LandmarkId(3),{9,0,2},error),"mixed-policy views");
    const auto oldest=window.marginalize(MarginalizationPolicy::MARGIN_OLD);
    require(oldest.success && oldest.priorLandmarks==0 && window.graph().prior,
        "oldest track elimination produces a state/calibration prior, without retained rho");
    require(!window.landmarks().find(LandmarkId(1)) && window.landmarks().find(LandmarkId(2))
        && window.landmarks().find(LandmarkId(3)),"OLD retires its own mature track, not other anchors");
    require(window.append(f.prediction(10),*f.frame(10),error),"post-prior redundancy append");
    require(window.addObservation(LandmarkId(2),{10,0,1},error)
        && window.addObservation(LandmarkId(2),{10,1,1},error)
        && window.addObservation(LandmarkId(3),{10,0,2},error),"post-prior views");
    const auto second=window.marginalize(MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(second.success && second.priorLandmarks==0 && second.resetPriorUnchanged && second.resetPriorProtectedBlocks>0
        && window.landmarks().find(LandmarkId(2))
        && window.valid(error),"prior-only pose elimination must not couple retained landmarks");
    SlidingWindowManager rear(options);
    for(auto block:makeRigCalibration(f.rig,kinematics::Transformation()).parameterBlocks())
        require(rear.addParameter(std::move(block),error),"rear reset rig");
    for(uint64_t i=1;i<=3;++i) require(rear.append(f.prediction(i),*f.frame(i),error),"rear reset append");
    require(rear.addLandmark(LandmarkId(1),{2,1,3},f.rho(2,1,3),error)
        && rear.addObservation(LandmarkId(1),{1,1,3},error)
        && rear.addObservation(LandmarkId(1),{3,1,3},error),"rear hemisphere reset support");
    const auto rearReport=rear.marginalize(MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(rearReport.success && rearReport.resetLandmarks.size()==1
        && rear.landmarks().find(LandmarkId(1))->anchorBearing.z()<0
        && rear.graph().blocks.at({BlockKind::Landmark,1}).value[0]>0,"negative z must keep positive inverse range after reset");
    Eigen::Vector3d rearWorld; require(rear.landmarkPoint(LandmarkId(1),rearWorld),"rear reset geometry");
    debug::recordError("rear_reset_world_error",(rearWorld-f.points[3]).norm(),1e-4);
}
void oldestFrameRetainsPriorFreeTracks() {
    Fixture f; BackendOptions options; options.windowSize=2;
    options.preserveOldestVisualInformation=false; // Retain explicit coverage of the opt-in approximate reset policy.
    SlidingWindowManager window(options); std::string error;
    for(auto block:makeRigCalibration(f.rig,kinematics::Transformation()).parameterBlocks())
        require(window.addParameter(std::move(block),error),"oldest retention rig");
    for(uint64_t id=1;id<=3;++id) require(window.append(f.prediction(id),*f.frame(id),error),"oldest retention append");
    require(window.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"oldest mature point");
    for(auto key:std::vector<KeypointIdentifier>{{2,0,0},{2,1,0},{3,0,0},{3,1,0}})
        require(window.addObservation(LandmarkId(1),key,error),"oldest retained support");
    require(window.addLandmark(LandmarkId(2),{1,0,1},f.rho(1,0,1),error)
        && window.addObservation(LandmarkId(2),{3,0,1},error),"oldest single-ray point");
    const auto result=window.marginalize(MarginalizationPolicy::MARGIN_OLD);
    require(result.success && result.resetLandmarks==std::vector<LandmarkId>{LandmarkId(1)}
        && result.demotedLandmarks==std::vector<LandmarkId>{LandmarkId(2)}
        && result.retiredLandmarks.empty() && result.priorLandmarks==0,"OLD retains supported IDs without a dense landmark prior");
    const auto point=window.landmark(LandmarkId(1));
    require(point && point->pointWorld && point->metadata.anchor.frameId==3
        && !window.landmarks().find(LandmarkId(2))->initialized,"oldest retained keyframe and single-ray seed");
    require(result.resetPriorUnchanged && point->metadata.resetCount==1 && point->metadata.initializationCount==1
        && window.landmarks().find(LandmarkId(2))->demotionCount==1,"lifecycle counters and frozen-prior guard commit together");
    debug::recordError("oldest_reset_world_point_error",(*point->pointWorld-f.points[0]).norm(),1e-4);
    require(!window.landmarks().find(KeypointIdentifier{1,0,0})
        && !window.landmarks().consumed({2,0,0}) && window.valid(error),
        "discarded anchor leaves no dangling dependency or falsely absorbed surviving measurement");
    require(window.append(f.prediction(4),*f.frame(4),error)
        && window.addObservation(LandmarkId(1),{4,0,0},error),"continue retained mature identity");
    const auto next=window.marginalize(MarginalizationPolicy::MARGIN_OLD);
    require(next.success && next.resetLandmarks.empty() && window.landmarks().find(LandmarkId(1))->resetCount==1,
        "newest retained keyframe avoids an immediate second reset when its predecessor leaves");
}
void exportReanchorAudit(const std::string& name,const ReanchorInformationReport& audit) {
    auto summary=debug::artifactFile(name+"_summary.csv");
    if(summary) summary<<"equivalent,point_error,point_derivative_error,information_error,gradient_error\n"
        <<audit.equivalent<<','<<audit.pointError<<','<<audit.pointDerivativeError<<','<<audit.informationError<<','<<audit.gradientError<<'\n';
    auto columns=debug::artifactFile(name+"_columns.csv");
    if(columns) {
        columns<<"block_kind,id,offset,tangent_size\n";
        for(size_t i=0;i<audit.keys.size();++i) columns<<int(audit.keys[i].kind)<<','<<audit.keys[i].id<<','<<audit.offsets[i]<<','<<audit.sizes[i]<<'\n';
    }
    const Eigen::MatrixXd transportedH=audit.oldFromNew.transpose()*audit.oldInformation*audit.oldFromNew;
    const Eigen::VectorXd transportedB=audit.oldFromNew.transpose()*audit.oldGradient;
    const Eigen::MatrixXd transportedPoint=audit.oldPointJacobian*audit.oldFromNew;
    auto geometry=debug::artifactFile(name+"_point_jacobian.csv");
    if(geometry) {
        geometry<<"row,column,transported_point_derivative,candidate_point_derivative,absolute_error\n";
        for(int r=0;r<3;++r) for(int c=0;c<transportedPoint.cols();++c)
            geometry<<r<<','<<c<<','<<transportedPoint(r,c)<<','<<audit.newPointJacobian(r,c)<<','
                <<std::abs(transportedPoint(r,c)-audit.newPointJacobian(r,c))<<'\n';
    }
    auto matrices=debug::artifactFile(name+"_information.csv");
    if(matrices) {
        matrices<<"row,column,old_from_new,transported_information,candidate_information,absolute_error\n";
        for(int r=0;r<transportedH.rows();++r) for(int c=0;c<transportedH.cols();++c)
            matrices<<r<<','<<c<<','<<audit.oldFromNew(r,c)<<','<<transportedH(r,c)<<','<<audit.newInformation(r,c)
                <<','<<std::abs(transportedH(r,c)-audit.newInformation(r,c))<<'\n';
    }
    auto gradient=debug::artifactFile(name+"_gradient.csv");
    if(gradient) {
        gradient<<"row,transported_gradient,candidate_gradient,absolute_error\n";
        for(int r=0;r<transportedB.size();++r) gradient<<r<<','<<transportedB[r]<<','<<audit.newGradient[r]<<','
            <<std::abs(transportedB[r]-audit.newGradient[r])<<'\n';
    }
}
void oldestSchurKeepsIdentityWithoutReusingEvidence() {
    Fixture f; BackendOptions options; options.windowSize=2;
    SlidingWindowManager window(options); std::string error;
    for(auto block:makeRigCalibration(f.rig,kinematics::Transformation()).parameterBlocks())
        require(window.addParameter(std::move(block),error),"Schur identity rig");
    for(uint64_t id=1;id<=3;++id) require(window.append(f.prediction(id),*f.frame(id),error),"Schur identity states");
    require(window.addFactor(makeParameterPrior(window.graph().blocks.at(BlockKey::pose(StateId(1))),
        10.0*Eigen::Matrix<double,6,6>::Identity())).accepted(),"Schur identity gauge");
    require(window.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error)
        && window.addObservation(LandmarkId(1),{2,0,0},error)
        && window.addObservation(LandmarkId(1),{3,1,0},error),"outgoing-anchor visual evidence");
    require(window.addLandmark(LandmarkId(2),{2,0,1},f.rho(2,0,1),error)
        && window.addObservation(LandmarkId(2),{1,1,1},error)
        && window.addObservation(LandmarkId(2),{3,0,1},error),"retained implicit anchor also participates in old evidence");
    auto referenceGraph=window.graph(); auto denseOptions=options; denseOptions.landmarkFirstMarginalization=false;
    const auto reference=Marginalizer(denseOptions).compute(referenceGraph,{BlockKey::pose(StateId(1)),
        BlockKey::speedBias(StateId(1)),{BlockKind::Landmark,1},{BlockKind::Landmark,2}});
    require(reference.success && reference.prior,"independent joint dense Schur reference");
    const auto result=window.marginalize(MarginalizationPolicy::MARGIN_OLD);
    require(result.success && result.schurPreservedTracks.size()==2 && result.scalarSchurLandmarks==2
        && result.retiredLandmarks.empty() && result.priorLandmarks==0,"OLD consumes visual information and retains appearance IDs");
    const auto actual=window.graph().prior;
    require(actual && actual->keys()==reference.prior->keys(),"Schur separators match dense reference");
    const auto& A=actual->squareRootInformation(); const auto& B=reference.prior->squareRootInformation();
    debug::recordDerivative("oldest_information_vs_joint_schur",(A.transpose()*A-B.transpose()*B).norm(),(B.transpose()*B).norm(),1e-6,1e-6);
    debug::recordDerivative("oldest_gradient_vs_joint_schur",
        (A.transpose()*actual->residualAtReference()-B.transpose()*reference.prior->residualAtReference()).norm(),
        (B.transpose()*reference.prior->residualAtReference()).norm(),1e-6,1e-6);
    require(!window.landmarks().find(LandmarkId(1))->initialized
        && window.landmarks().consumed({2,0,1})
        && window.landmarks().find(KeypointIdentifier{2,0,1})->use==ObservationUse::Absorbed,
        "implicit retained anchor is absorbed even without a standalone residual ID");
    require(!window.initializeLandmark(LandmarkId(1),f.rho(3,1),error),"old absorbed views cannot initialize another range");
    require(window.append(f.prediction(4),*f.frame(4),error)
        && window.addObservation(LandmarkId(1),{4,0,0},error)
        && window.addObservation(LandmarkId(1),{4,1,0},error),"fresh stereo observations keep the SAME identity");
    require(!window.promotionWouldBeMarginalized({4,0,0},{{2,0,0},{4,0,0},{4,1,0}}),
        "absorbed outgoing support cannot defer fresh stereo promotion");
    require(window.initializeLandmark(LandmarkId(1),f.rho(4,0),error),"fresh stereo can mature the SAME identity");
    size_t live=0;
    for(const auto& key:window.landmarks().ofLandmark(LandmarkId(1))) {
        const auto observation=window.landmarks().find(key);
        if(observation->use==ObservationUse::LiveFactor) { ++live; require(key.frameId==4,"only fresh views have live costs"); }
    }
    require(live==1 && window.landmarks().find(LandmarkId(1))->initializationCount==2
        && window.graph().prior==actual && window.valid(error),"reinitialization preserves frozen prior and consumption history");
    const auto retained=window.landmarks().find(KeypointIdentifier{4,1,0})->factorId;
    const auto retainedCost=window.graph().factors.at(retained).cost;
    const auto next=window.marginalize(MarginalizationPolicy::MARGIN_OLD);
    require(next.success && next.ignoredAbsorbedIncidentObservations>0 && next.liveIncidentRanges==0
        && next.sameFrameRangeEliminations==0 && next.schurPreservedTracks.empty(),
        "absorbed frame-2 rays must not trigger a new rho's Schur elimination");
    require(window.landmarks().find(LandmarkId(1))->initialized && window.graph().factors.at(retained).cost==retainedCost
        && window.landmarks().find(LandmarkId(1))->initializationCount==2 && window.valid(error),
        "fresh mature range and its live stereo factor survive consumed-only incidence");
}
void visualPoseCacheMatchesUncachedTrials() {
    Fixture f; FixedLagBackend backend; initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    ObservationOptions anchorTime,observerTime;
    WindowObservation anchor,observer;
    require(backend.window().frames().observation({1,0,0},1,anchor,error)
        && backend.window().frames().observation({2,1,0},1,observer,error),"cache fixture bearings");
    anchorTime.bearingVelocity=anchor.bearing.cross(Eigen::Vector3d(0.12,0.04,0));
    observerTime.bearingVelocity=observer.bearing.cross(Eigen::Vector3d(-0.08,0.05,0));
    anchorTime.referenceTimeOffset=0.003; observerTime.referenceTimeOffset=-0.002;
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error,anchorTime)
        && backend.addObservation(LandmarkId(1),{2,1,0},error,observerTime),"cache mixed-camera factor");
    auto graph=backend.window().graph(); const auto observation=backend.window().landmarks().find(KeypointIdentifier{2,1,0});
    const auto& factor=graph.factors.at(observation->factorId); std::vector<const double*> bindings;
    for(auto key:factor.keys) bindings.push_back(graph.blocks.at(key).value.data());
    VisualEvaluationCache cache; const auto cached=cacheWindowVisualCost(*factor.cost,bindings,cache);
    require(bool(cached),"native sphere cost supports solver-local pose cache");
    require(cache.cameraCount()==2,"Two state/camera chains must be shared by the pair");
    require(cache.pairCount()==1,"One camera-pair transform is bound for the mixed-camera factor");
    const auto duplicate=cacheWindowVisualCost(*factor.cost,bindings,cache);
    require(bool(duplicate) && cache.pairCount()==1,"Repeated factors must share the same camera-pair transform");
    require(cache.sharedTimedAnchorCount()==1,"Repeated factors share one moving anchor");
    const BlockKey td{BlockKind::CameraTimeOffset,0};
    const size_t tdIndex=std::find(factor.keys.begin(),factor.keys.end(),td)-factor.keys.begin();
    require(tdIndex<factor.keys.size(),"fixture exposes camera time offset");

    for(int trial=0;trial<3;++trial) {
        graph.blocks.at(td).value[0]=trial==0?0.003:0.004;
        if(trial) {
            auto& pose=graph.blocks.at(trial==1?BlockKey::pose(StateId(2)):BlockKey{BlockKind::CameraExtrinsic,0});
            Eigen::Matrix<double,6,1> step; step<<0.001,-0.002,0.003,0.002,-0.001,0.003;
            Eigen::VectorXd next(7); require(pose.manifold->Plus(pose.value.data(),step.data(),next.data()),"cache trial perturbation");
            std::copy(next.data(),next.data()+7,pose.value.data());
        }
        std::vector<Eigen::VectorXd> input,ja,jb; std::vector<const double*> x; std::vector<double*> pa,pb;
        for(auto key:factor.keys) { input.push_back(graph.blocks.at(key).value); ja.emplace_back(2*input.back().size()); jb.emplace_back(2*input.back().size()); }
        // Deliberately pass different addresses, as Ceres may do internally.
        for(size_t i=0;i<input.size();++i) { x.push_back(input[i].data()); pa.push_back(ja[i].data()); pb.push_back(jb[i].data()); }
        cache.PrepareForEvaluation(false,true); Eigen::Vector2d a,b;
        require(factor.cost->Evaluate(x.data(),a.data(),nullptr) && cached->Evaluate(x.data(),b.data(),nullptr),"cache residual trial");
        debug::recordError("cached_residual_"+std::to_string(trial),(a-b).norm(),1e-10);
        cache.PrepareForEvaluation(true,false);
        require(factor.cost->Evaluate(x.data(),a.data(),pa.data()) && cached->Evaluate(x.data(),b.data(),pb.data()),"cached derivative trial");
        for(size_t i=0;i<ja.size();++i) debug::recordDerivative("cached_jacobian_"+std::to_string(trial)+"_"+std::to_string(i),
            (ja[i]-jb[i]).norm(),ja[i].norm(),1e-10,1e-10);
        const auto refreshes=cache.refreshCount(); cache.PrepareForEvaluation(true,false);
        require(cache.refreshCount()==refreshes,"Jacobian request at the same evaluation point must reuse pair data");
        for(size_t requested=0;requested<jb.size();++requested) {
            auto partial=pb; for(size_t i=0;i<partial.size();++i) if(i!=requested) partial[i]=nullptr;
            jb[requested].setConstant(std::numeric_limits<double>::quiet_NaN());
            require(cached->Evaluate(x.data(),b.data(),partial.data()),"Cached partial derivatives including rho-only must evaluate");
            debug::recordDerivative("cached_partial_"+std::to_string(trial)+"_"+std::to_string(requested),
                (ja[requested]-jb[requested]).norm(),ja[requested].norm(),1e-10,1e-10);
        }
        VisualEvaluationCache fixedCache;
        const auto fixed=cacheWindowVisualCost(*factor.cost,bindings,fixedCache,true,true);
        fixedCache.PrepareForEvaluation(true,true);
        auto fixedA=pa,fixedB=pb;
        fixedA[tdIndex]=nullptr; fixedB[tdIndex]=nullptr;
        require(factor.cost->Evaluate(x.data(),a.data(),fixedA.data())
            && fixed->Evaluate(x.data(),b.data(),fixedB.data()),"fixed-time compensation and basis");
        debug::recordError("fixed_time_residual_"+std::to_string(trial),(a-b).norm(),1e-10);
        for(size_t i=0;i<ja.size();++i) if(i!=tdIndex)
            debug::recordDerivative("fixed_time_jacobian_"+std::to_string(trial)+"_"+std::to_string(i),
                (ja[i]-jb[i]).norm(),ja[i].norm(),1e-10,1e-10);
    }
}
void cachedVisualGateMatchesReference() {
    Fixture f; FixedLagBackend backend; initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error)
        && backend.addObservation(LandmarkId(1),{2,1,0},error),"gate cache fixture");
    auto original=backend.window().graph();
    for(auto& p:original.blocks) p.second.constant=p.first.kind!=BlockKind::Landmark;
    original.blocks.at({BlockKind::Landmark,1}).value[0]*=1.01;
    auto a=original,b=original; BackendOptions reference,optimized;
    reference.cacheVisualPairs=false; optimized.cacheVisualPairs=true;
    const auto ra=BackendOptimizer(reference).optimize(a),rb=BackendOptimizer(optimized).optimize(b);
    require(ra.accepted && rb.accepted,"both visual gate/cache paths must solve");
    require(ra.preRejected==rb.preRejected && ra.postRejected==rb.postRejected,"cache must preserve NIS decisions");
    debug::recordError("gate_cache_rho",std::abs(a.blocks.at({BlockKind::Landmark,1}).value[0]-b.blocks.at({BlockKind::Landmark,1}).value[0]),1e-8);
    debug::recordError("gate_cache_final_nis",std::abs(ra.postSolveResiduals[0].squaredNorm-rb.postSolveResiduals[0].squaredNorm),1e-8);
    require(rb.visualCachedCameras==2,"Ceres and gate use shared state/camera geometry");
}
void rigRotationSolvePreservesTranslationAndPriorCoordinates() {
    const BlockKey rigKey{BlockKind::CameraExtrinsic,0};
    const BlockKey tdKey{BlockKind::CameraTimeOffset,0};
    const Eigen::Vector3d position(0.15,-0.07,0.04);
    const Eigen::Quaterniond targetRotation=expSO3(Eigen::Vector3d(0.030,-0.010,0.005));
    BackendGraph graph;
    graph.blocks.emplace(rigKey, ParameterBlock::pose(rigKey,
        kinematics::Transformation(position,Eigen::Quaterniond::Identity())));
    graph.blocks.emplace(tdKey, ParameterBlock::euclidean(tdKey,Eigen::VectorXd::Zero(1)));
    auto& td=graph.blocks.at(tdKey);
    td.lower=Eigen::VectorXd::Constant(1,-0.004);
    td.upper=Eigen::VectorXd::Constant(1,0.004);
    const auto rigReference=ParameterBlock::pose(rigKey,kinematics::Transformation(position,targetRotation));
    graph.prior=std::make_shared<MarginalizationPrior>(std::vector<ParameterBlock>{rigReference},
        Eigen::MatrixXd::Identity(6,6),Eigen::VectorXd::Zero(6));
    const auto tdReference=ParameterBlock::euclidean(tdKey,Eigen::VectorXd::Constant(1,0.006));
    auto timePrior=makeParameterPrior(tdReference,Eigen::MatrixXd::Constant(1,1,1000),FactorKind::Calibration);
    timePrior.id=1; timePrior.lifecycle=FactorLifecycle::Active;
    graph.factors.emplace(timePrior.id,std::move(timePrior));

    SolveBudget budget; budget.cameraRotationOnly=true; budget.maxIterations=15; budget.timeSeconds=0.5;
    const BackendOptimizer optimizer;
    auto retryGraph=graph;
    const auto step=optimizer.optimize(graph,{},budget);
    require(step.accepted,"bounded clock and rotation-only solve must succeed");
    const auto& rigBlock=graph.blocks.at(rigKey);
    const auto rig=blockPose(rigBlock);
    require(rigBlock.tangentSize()==6 && graph.prior->tangentSizes().front()==6,
        "rotation-only solve preserves the six-coordinate FEJ prior");
    debug::recordError("restricted_rig_translation",(rig.r()-position).norm(),1e-12);
    debug::recordError("restricted_rig_rotation",logSO3(rig.q()*targetRotation.conjugate()).norm(),1e-6);
    debug::recordError("camera_td_bound",std::abs(graph.blocks.at(tdKey).value[0]-0.004),1e-8);

    budget.calibrationBlockViolation=[tdKey](const ParameterBlock& block)->std::string {
        if (block.key==tdKey && block.value[0]>0.002) return "test clock outside physical range";
        return {};
    };
    const auto rejected=optimizer.optimize(retryGraph,{},budget);
    require(!rejected.accepted && rejected.rejectedCalibration==std::vector<BlockKey>{tdKey}
        && retryGraph.blocks.at(tdKey).value[0]==0,"identify only the clock block and restore the transaction");
    retryGraph.blocks.at(tdKey).constant=true;
    const auto retried=optimizer.optimize(retryGraph,{},budget);
    require(retried.accepted && retried.rejectedCalibration.empty() && retryGraph.blocks.at(tdKey).value[0]==0,
        "refit with the clock fixed must keep the rotation eligible for acceptance");
    debug::recordError("selective_retry_rotation",logSO3(blockPose(retryGraph.blocks.at(rigKey)).q()*targetRotation.conjugate()).norm(),1e-6);
}
void constraintGradientBalanceIsRobustAndReadOnly() {
    const auto insert=[](BackendGraph& graph,FactorRecord f) {
        f.id=graph.factors.size()+1; f.lifecycle=FactorLifecycle::Active; graph.factors.emplace(f.id,std::move(f));
    };
    BackendGraph graph; const BlockKey key{BlockKind::Auxiliary,120};
    auto block=ParameterBlock::euclidean(key,Eigen::VectorXd::Constant(1,1.0)); graph.blocks.emplace(key,block);
    auto ref=block; ref.value[0]=0;
    auto visual=makeParameterPrior(ref,Eigen::MatrixXd::Identity(1,1)); visual.kind=FactorKind::Visual;
    visual.loss=std::make_shared<::ceres::HuberLoss>(0.5); insert(graph,visual);
    ref.value[0]=2; auto imu=makeParameterPrior(ref,Eigen::MatrixXd::Identity(1,1)); imu.kind=FactorKind::Imu; insert(graph,imu);
    ref.value[0]=3; graph.prior=std::make_shared<MarginalizationPrior>(std::vector<ParameterBlock>{ref},
        Eigen::MatrixXd::Identity(1,1),Eigen::VectorXd::Zero(1));
    ref.value[0]=100; auto rejected=makeParameterPrior(ref,Eigen::MatrixXd::Identity(1,1)); insert(graph,rejected);
    graph.factors.rbegin()->second.lifecycle=FactorLifecycle::Rejected;
    const auto prior=graph.prior; const auto report=BackendOptimizer().constraintGradients(graph,{key});
    require(report.success && report.factors[0]==1 && report.factors[1]==1 && report.factors[3]==1
        && report.factors[4]==0,"gradient audit categorizes accepted costs and excludes rejected evidence");
    debug::recordError("robust_visual_gradient",std::abs(report.gradients[0][0]-0.5),1e-10);
    debug::recordError("imu_gradient",std::abs(report.gradients[1][0]+1),1e-10);
    debug::recordError("prior_gradient",std::abs(report.gradients[3][0]+2),1e-10);
    const auto objective=[](double x) { return 0.5*(x-0.25)+0.5*(x-2)*(x-2)+0.5*(x-3)*(x-3); };
    const double numeric=(objective(1+1e-6)-objective(1-1e-6))/2e-6;
    debug::recordError("total_gradient_vs_numeric",std::abs(report.gradients[0][0]+report.gradients[1][0]
        +report.gradients[3][0]-numeric),1e-7);
    require(graph.blocks.at(key).value[0]==1 && graph.prior==prior
        && graph.factors.rbegin()->second.lifecycle==FactorLifecycle::Rejected,"audit preserves values, prior and lifecycle");
    Fixture f; FixedLagBackend backend; initialize(backend,f); append(backend,f,1); append(backend,f,2); std::string error;
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error)
        && backend.addObservation(LandmarkId(1),{2,1,0},error),"gradient audit real sphere factor");
    auto native=backend.window().graph(); const auto& record=native.factors.at(
        backend.window().landmarks().find(KeypointIdentifier{2,1,0})->factorId);
    auto& pose=native.blocks.at(BlockKey::pose(StateId(2))); pose.value[0]+=0.03;
    const auto actual=BackendOptimizer().constraintGradients(native,record.keys); LinearizedSystem reference;
    require(actual.success && Marginalizer(BackendOptions()).linearize(native,{record.id},false,true,reference,error),
        "left-tangent gradient reference uses Ceres robust manifold evaluation");
    auto csv=debug::artifactFile("constraint_gradients.csv");
    if(csv) csv<<"block_kind,id,component,audit,ceres_reference\n";
    for(size_t i=0;i<actual.keys.size();++i) {
        const auto found=std::find(reference.keys.begin(),reference.keys.end(),actual.keys[i]);
        require(found!=reference.keys.end(),"reference gradient block exists");
        const auto index=static_cast<size_t>(found-reference.keys.begin());
        if(csv) for(int c=0;c<actual.sizes[i];++c) csv<<int(actual.keys[i].kind)<<','<<actual.keys[i].id<<','<<c<<','
            <<actual.gradients[0][actual.offsets[i]+c]<<','<<reference.b[reference.offsets[index]+c]<<'\n';
        debug::recordDerivative("left_tangent_gradient_"+std::to_string(i),
            (actual.gradients[0].segment(actual.offsets[i],actual.sizes[i])
             -reference.b.segment(reference.offsets[index],reference.sizes[index])).norm(),
            reference.b.segment(reference.offsets[index],reference.sizes[index]).norm(),1e-8,1e-8);
    }
}
void reanchorInformationAudit() {
    Fixture f; FixedLagBackend backend; initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"audit anchor");
    require(backend.addObservation(LandmarkId(1),{2,1,0},error),"audit target");
    auto before=backend.window().graph();
    const auto landmark=*backend.window().landmarks().find(LandmarkId(1));
    const auto anchor=*backend.window().landmarks().find(landmark.anchor);
    const BlockKey rho{BlockKind::Landmark,1};
    std::vector<ParameterBlock> refs{before.blocks.at(BlockKey::pose(StateId(2))),before.blocks.at(rho)};
    Eigen::MatrixXd A=Eigen::MatrixXd::Zero(5,7);
    for(int i=0;i<A.rows();++i) for(int j=0;j<A.cols();++j) A(i,j)=0.2*std::sin(1.3+i+0.7*j)+(i==j?1:0);
    const Eigen::VectorXd r0=Eigen::VectorXd::LinSpaced(5,-0.2,0.3);
    before.prior=std::make_shared<MarginalizationPrior>(refs,A,r0);
    ReanchorInformationReport identity;
    require(checkReanchorInformation(landmark,anchor,landmark,anchor,backend.window().frames(),before,before,
        BackendOptions(),identity,error),"identity chart audit must evaluate");
    exportReanchorAudit("identity",identity);
    require(identity.equivalent,"unchanged physical chart and historical prior must be equivalent");
    debug::recordError("identity_information_error",identity.informationError,1e-10);
    debug::recordError("identity_gradient_error",identity.gradientError,1e-10);
    require(identity.oldGradient.norm()>1e-3,"nonzero prior gradient prevents a vacuous zero-gradient comparison");
    auto corrupted=before;
    corrupted.prior=std::make_shared<MarginalizationPrior>(refs,1.2*A,r0);
    ReanchorInformationReport bad;
    require(checkReanchorInformation(landmark,anchor,landmark,anchor,backend.window().frames(),before,corrupted,
        BackendOptions(),bad,error),"corrupted-prior audit must evaluate");
    exportReanchorAudit("corrupted_prior",bad);
    require(!bad.equivalent && bad.informationError>1e-3 && bad.gradientError>1e-3,
        "audit must detect historical information/gradient changes even at the same world point");
    debug::recordError("corrupted_prior_was_accepted",bad.equivalent?1:0,0);
    require(backend.window().graph().prior==nullptr,"audit must not mutate the live graph/prior");
}
void reanchorFixedBearingChangeIsRejected() {
    Fixture f; BackendOptions options; options.windowSize=2; options.maxSolverTimeSeconds=10;
    options.preserveSchurTrackIds=false; // This case explicitly checks retirement, not appearance-only ID retention.
    options.autoReanchorLandmarks=true;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    for(uint64_t i=1;i<=3;++i) { append(backend,f,i); poseConstraints(backend,StateId(i)); }
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"fixed-bearing anchor");
    require(backend.addObservation(LandmarkId(1),{2,0,0},error),"fixed-bearing next view");
    require(backend.addObservation(LandmarkId(1),{3,1,0},error),"fixed-bearing third view");
    const auto before=backend.window().graph(); auto candidate=before;
    const auto oldLandmark=*backend.window().landmarks().find(LandmarkId(1));
    const auto oldAnchor=*backend.window().landmarks().find(oldLandmark.anchor);
    const KeypointIdentifier key{2,0,0}; const auto newAnchor=*backend.window().landmarks().find(key);
    const auto estimate=backend.window().landmark(LandmarkId(1));
    require(estimate && estimate->pointWorld,"audit requires an initialized world point");
    const Eigen::Vector3d world=*estimate->pointWorld;
    kinematics::Transformation camera;
    require(backend.window().frames().cameraPose(before,StateId(2),0,camera,error),"new audit camera");
    auto replacement=oldLandmark; replacement.anchor=key;
    replacement.anchorBearing=(camera.C().transpose()*(world-camera.r())).normalized();
    const BlockKey rho{BlockKind::Landmark,1};
    candidate.blocks.at(rho).value[0]=1/(world-camera.r()).norm(); candidate.blocks.at(rho).owner=StateId(2);
    for(const auto& observed:backend.window().landmarks().ofLandmark(LandmarkId(1))) {
        const auto& observation=*backend.window().landmarks().find(observed); FactorRecord factor;
        require(makeWindowVisualFactor(replacement,newAnchor,observation,backend.window().frames(),candidate,options,factor,error),"build audit candidate");
        if(observation.use==ObservationUse::LiveFactor) {
            const auto& old=candidate.factors.at(observation.factorId);
            factor.id=old.id; factor.lifecycle=old.lifecycle; factor.loss=old.loss; factor.nisThreshold=old.nisThreshold;
            candidate.factors.at(factor.id)=std::move(factor);
        } else {
            factor.id=candidate.factors.rbegin()->first+1; factor.lifecycle=FactorLifecycle::Active;
            candidate.factors.emplace(factor.id,std::move(factor));
        }
    }
    ReanchorInformationReport audit;
    require(checkReanchorInformation(oldLandmark,oldAnchor,replacement,newAnchor,backend.window().frames(),before,candidate,options,audit,error),
        "fixed-bearing information audit must evaluate");
    exportReanchorAudit("fixed_bearing_candidate",audit);
    debug::recordError("nominal_point_error",audit.pointError,1e-8);
    require(!audit.equivalent && audit.pointDerivativeError>1e-3,
        "equal nominal points on different scalar anchor rays are not equivalent perturbation charts");
    debug::recordError("non_equivalent_chart_was_accepted",audit.equivalent?1:0,0);
    const auto result=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(result.accepted && result.marginalization->reanchoredLandmarks.empty()
        && !backend.window().landmarks().find(LandmarkId(1)),"opt-in unsafe chart must fall back to Schur retirement");
    require(backend.window().valid(error),"rejected chart must preserve domain/prior consistency");
}
void reanchorFullCoordinateTransportReference() {
    // Positive control with a genuinely bijective 3D landmark chart. This verifies
    // coupled pose/landmark information transport; it does NOT claim that the
    // production fixed-bearing scalar chart has acquired two extra coordinates.
    const kinematics::Transformation oldCamera(Eigen::Vector3d(0.2,-0.1,0.3),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.2,Eigen::Vector3d::UnitY())));
    const kinematics::Transformation newCamera(Eigen::Vector3d(0.6,0.2,0.1),
        Eigen::Quaterniond(Eigen::AngleAxisd(-0.15,Eigen::Vector3d::UnitX())));
    const Eigen::Vector3d localOld(0.5,-0.2,4),world=oldCamera.C()*localOld+oldCamera.r();
    const Eigen::Vector3d localNew=newCamera.C().transpose()*(world-newCamera.r());
    const LeftSE3Manifold manifold;
    const auto perturbed=[&](const kinematics::Transformation& camera,const Eigen::VectorXd& delta) {
        Eigen::Matrix<double,7,1> input,output; input<<camera.r(),camera.q().coeffs();
        require(manifold.Plus(input.data(),delta.data(),output.data()),"reference left perturbation");
        return kinematics::Transformation(Eigen::Vector3d(output.head<3>()),Eigen::Quaterniond(output[6],output[3],output[4],output[5]));
    };
    const auto map=[&](const Eigen::VectorXd& z) {
        const auto a=perturbed(oldCamera,z.head(6)),b=perturbed(newCamera,z.segment(6,6));
        const Eigen::Vector3d p=b.C()*(localNew+z.tail<3>())+b.r();
        Eigen::VectorXd x=z; x.tail<3>()=a.C().transpose()*(p-a.r())-localOld; return x;
    };
    Eigen::MatrixXd G=Eigen::MatrixXd::Identity(15,15);
    G.block<3,3>(12,0)=-oldCamera.C().transpose(); G.block<3,3>(12,3)=oldCamera.C().transpose()*skew(world);
    G.block<3,3>(12,6)=oldCamera.C().transpose(); G.block<3,3>(12,9)=-oldCamera.C().transpose()*skew(world);
    G.block<3,3>(12,12)=oldCamera.C().transpose()*newCamera.C();
    Eigen::MatrixXd numeric(15,15);
    for(int c=0;c<15;++c) {
        Eigen::VectorXd p=Eigen::VectorXd::Zero(15),m=p; p[c]=1e-6; m[c]=-1e-6;
        numeric.col(c)=(map(p)-map(m))/2e-6;
    }
    debug::recordDerivative("full_chart_transport_jacobian",(G-numeric).norm(),numeric.norm(),1e-6,1e-9);
    Eigen::MatrixXd A(20,15);
    for(int r=0;r<A.rows();++r) for(int c=0;c<A.cols();++c) A(r,c)=(r==c?1.0:0)+0.03*std::sin(0.7*r+1.3*c);
    const Eigen::VectorXd r0=Eigen::VectorXd::LinSpaced(20,-0.3,0.5); const Eigen::MatrixXd transported=A*G;
    const BlockKey a{BlockKind::Auxiliary,1},b{BlockKind::Auxiliary,2},p{BlockKind::Auxiliary,3};
    BackendGraph oldGraph,newGraph;
    oldGraph.blocks.emplace(a,ParameterBlock::euclidean(a,Eigen::VectorXd::Zero(6)));
    oldGraph.blocks.emplace(b,ParameterBlock::euclidean(b,Eigen::VectorXd::Zero(6)));
    oldGraph.blocks.emplace(p,ParameterBlock::euclidean(p,Eigen::VectorXd::Zero(3))); newGraph=oldGraph;
    std::vector<ParameterBlock> references{oldGraph.blocks.at(a),oldGraph.blocks.at(b),oldGraph.blocks.at(p)};
    oldGraph.prior=std::make_shared<MarginalizationPrior>(references,A,r0);
    newGraph.prior=std::make_shared<MarginalizationPrior>(references,transported,r0);
    BackendOptions settings; Marginalizer marginalizer(settings);
    const auto oldResult=marginalizer.compute(oldGraph,{a,p}),newResult=marginalizer.compute(newGraph,{a,p});
    require(oldResult.success && newResult.success && oldResult.prior && newResult.prior,"transported priors must support Schur");
    require(oldResult.prior->keys()==newResult.prior->keys(),"same retained physical coordinates after Schur");
    const Eigen::MatrixXd oldH=oldResult.prior->squareRootInformation().transpose()*oldResult.prior->squareRootInformation();
    const Eigen::MatrixXd newH=newResult.prior->squareRootInformation().transpose()*newResult.prior->squareRootInformation();
    const Eigen::VectorXd oldB=oldResult.prior->squareRootInformation().transpose()*oldResult.prior->residualAtReference();
    const Eigen::VectorXd newB=newResult.prior->squareRootInformation().transpose()*newResult.prior->residualAtReference();
    debug::recordError("transported_schur_information_error",(oldH-newH).norm()/std::max(1.0,oldH.norm()),1e-10);
    debug::recordError("transported_schur_gradient_error",(oldB-newB).norm()/std::max(1.0,oldB.norm()),1e-10);
    auto schur=debug::artifactFile("transported_schur.csv");
    if(schur) {
        schur<<"row,column,original_information,transported_information,absolute_error\n";
        for(int r=0;r<oldH.rows();++r) for(int c=0;c<oldH.cols();++c)
            schur<<r<<','<<c<<','<<oldH(r,c)<<','<<newH(r,c)<<','<<std::abs(oldH(r,c)-newH(r,c))<<'\n';
    }
    auto csv=debug::artifactFile("full_coordinate_transport.csv");
    if(csv) {
        csv<<"row,column,analytic_old_from_new,numeric_old_from_new,absolute_error\n";
        for(int r=0;r<15;++r) for(int c=0;c<15;++c) csv<<r<<','<<c<<','<<G(r,c)<<','<<numeric(r,c)<<','<<std::abs(G(r,c)-numeric(r,c))<<'\n';
    }
    auto costs=debug::artifactFile("transported_prior_costs.csv");
    if(costs) costs<<"sample,old_linear_delta_cost,transported_delta_cost,absolute_error,nonlinear_chart_residual_remainder\n";
    for(int sample=0;sample<10;++sample) {
        Eigen::VectorXd z(15); for(int i=0;i<15;++i) z[i]=1e-6*std::sin(sample+0.4*i);
        const Eigen::VectorXd x=G*z,physical=map(z);
        const double* oldParameters[]{x.data(),x.data()+6,x.data()+12};
        const double* newParameters[]{z.data(),z.data()+6,z.data()+12};
        const double* physicalParameters[]{physical.data(),physical.data()+6,physical.data()+12};
        Eigen::VectorXd oldResidual(20),newResidual(20),physicalResidual(20);
        require(oldGraph.prior->Evaluate(oldParameters,oldResidual.data(),nullptr)
            && newGraph.prior->Evaluate(newParameters,newResidual.data(),nullptr)
            && oldGraph.prior->Evaluate(physicalParameters,physicalResidual.data(),nullptr),"evaluate frozen transported priors");
        const double oldCost=0.5*oldResidual.squaredNorm()-0.5*r0.squaredNorm();
        const double newCost=0.5*newResidual.squaredNorm()-0.5*r0.squaredNorm();
        const double remainder=(physicalResidual-newResidual).norm();
        if(costs) costs<<sample<<','<<oldCost<<','<<newCost<<','<<std::abs(oldCost-newCost)<<','<<remainder<<'\n';
        debug::recordError("transported_prior_delta_cost_"+std::to_string(sample),std::abs(oldCost-newCost),1e-10);
        debug::recordError("nonlinear_chart_second_order_remainder_"+std::to_string(sample),remainder,1e-9);
    }
}
void windowPoliciesAndRollback() {
    Fixture f; BackendOptions options; options.windowSize=2; options.maxSolverTimeSeconds=3.0;
    options.preserveSchurTrackIds=false;
    require(!options.autoReanchorLandmarks,"production defaults use information-preserving Schur retirement");
    options.autoReanchorLandmarks=true; // Unsafe chart changes must fail the information audit.
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    for(uint64_t id=1;id<=3;++id) { append(backend,f,id); poseConstraints(backend,StateId(id)); }
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"outgoing anchor");
    require(backend.addObservation(LandmarkId(1),{2,0,0},error),"temporal observation");
    require(backend.addObservation(LandmarkId(1),{3,1,0},error),"surviving cross-camera observation");
    auto before=backend.window().landmark(LandmarkId(1));
    const auto rejecting=[](StateId,BackendGraph&,std::string& error) { error="test preparation failure"; return false; };
    auto failed=backend.step(MarginalizationPolicy::MARGIN_OLD,{}, {}, {},rejecting);
    require(!failed.retiredState && failed.retiredVisualPoints.empty(),"failed transaction emits no accepted history receipt");
    require(failed.optimizedVisualHistory.empty() && failed.rejectedVisualObservations.empty(),"failed transaction cannot leak global archive evidence");
    require(!failed.accepted && backend.window().size()==3 && !backend.snapshot(),"failed marginalization rolls back graph and domain");
    require(backend.window().landmarks().find(LandmarkId(1))->anchor.frameId==1,"anchor metadata rolled back");
    auto accepted=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(accepted.accepted && accepted.marginalization->reanchoredLandmarks.empty(),"non-equivalent scalar chart is not committed");
    require(accepted.marginalization->retiredMature==1 && accepted.marginalization->retiredMatureWithLiveViews==1
        && accepted.marginalization->retirements.front().reason=="information_mismatch",
        "retirement receipt must report rejected chart and live support before Schur absorption");
    auto after=backend.window().landmark(LandmarkId(1));
    require(before && !after,"unsafe outgoing scalar chart is Schur-retired");
    require(!backend.window().frames().measurements(StateId(1)) && backend.window().landmarks().inFrame(StateId(1)).empty(),"frame payload and reverse index evicted");
    require(accepted.snapshot->landmarks.empty(),"accepted snapshot excludes retired chart");
    const auto published=backend.snapshot();
    auto rejected=backend.step(std::nullopt,{},[](const BackendGraph&,std::string& error){error="test quality failure";return false;});
    require(!rejected.accepted && backend.snapshot()==published && backend.window().valid(error),"whole-domain solve rollback");
    // A non-prior track with only one surviving ray keeps its ID as a seed.
    options.autoReanchorLandmarks=false;
    FixedLagBackend second(options); initialize(second,f);
    for(uint64_t id=1;id<=3;++id) { append(second,f,id); poseConstraints(second,StateId(id)); }
    require(second.addLandmark(LandmarkId(1),{2,0,0},f.rho(2,0),error),"second-new anchor");
    require(second.addObservation(LandmarkId(1),{3,1,0},error),"second-new observation");
    auto removed=second.step(MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(removed.retiredState && removed.retiredState->id==StateId(2)
        && removed.retiredVisualPoints.empty(),"seed demotion is not a retired visual point export");
    require(removed.accepted && removed.marginalization->removed==StateId(2),"second-newest policy");
    require(second.window().landmarks().find(LandmarkId(1)) && !second.window().landmarks().find(LandmarkId(1))->initialized
        && second.window().landmarks().find(LandmarkId(1))->anchor.frameId==3
        && removed.marginalization->demotedLandmarks==std::vector<LandmarkId>{LandmarkId(1)}
        && !second.window().landmarks().consumed({3,1,0}),"one-ray track retains its identity without a fictitious range constraint");
    require(!second.addLandmark(LandmarkId(2),{3,1,0},f.rho(3,1),error),"retained seed observation cannot be assigned to another track");
    require(second.window().order().front()==StateId(1) && second.window().order().back()==StateId(3),"retained frame membership");
    require(!removed.mapCandidate,"second-newest does not commit map");
    append(second,f,4);
    require(second.addObservation(LandmarkId(1),{4,1,0},error)
        && second.initializeLandmark(LandmarkId(1),f.rho(3,1),error),"demoted ID can recover range from new real multi-view support");
    require(second.window().landmarks().find(LandmarkId(1))->initialized
        && second.window().graph().blocks.at({BlockKind::Landmark,1}).owner==StateId(3)
        && second.window().valid(error),"reinitialized track uses its new anchor owner and original ID");
}
void tenFrameWindowUsesOneInsertionSlot() {
    Fixture fixture; BackendOptions defaults;
    require(defaults.windowSize==10,"Default retained window must contain ten frame states");
    defaults.maxSolverTimeSeconds=10;
    FixedLagBackend backend(defaults); initialize(backend,fixture); std::string error;
    for(uint64_t id=1;id<=10;++id) { append(backend,fixture,id); poseConstraints(backend,StateId(id)); }
    require(backend.window().size()==10 && !backend.window().needsMarginalization(),"Ten retained frames must fit without eviction");
    append(backend,fixture,11); poseConstraints(backend,StateId(11));
    require(backend.window().size()==11 && backend.window().needsMarginalization(),"New frame must occupy exactly one temporary slot");
    const auto extra=fixture.frame(12);
    require(!backend.append(fixture.prediction(12),*extra,{},error) && backend.window().size()==11,"A second insertion must wait for marginalization");
    const auto oldest=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(oldest.accepted && oldest.snapshot->states.size()==10 && oldest.retiredState->id==StateId(1)
        && backend.window().order().front()==StateId(2) && backend.window().order().back()==StateId(11),"OLD must restore ten retained frames");
    const auto exportStates=[](const std::string& policy,const BackendSnapshot& snapshot) {
        debug::recordError(policy+"_state_count_error",std::abs(double(snapshot.states.size())-10),0);
        auto csv=debug::artifactFile(policy+"_states.csv");
        if(csv) {
            csv<<"frame,timestamp_ns,x,y,z,keyframe\n";
            for(const auto& state:snapshot.states) csv<<state.id.value()<<','<<state.timestamp.toNSec()<<','
                <<state.T_WB.r().x()<<','<<state.T_WB.r().y()<<','<<state.T_WB.r().z()<<','<<int(state.keyframe)<<'\n';
        }
    };
    exportStates("old",*oldest.snapshot);
    append(backend,fixture,12); poseConstraints(backend,StateId(12));
    const auto secondNewest=backend.step(MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(secondNewest.accepted && secondNewest.snapshot->states.size()==10 && secondNewest.retiredState->id==StateId(11)
        && backend.window().order().front()==StateId(2) && backend.window().order().back()==StateId(12),"SECOND_NEW must retain newest frame and restore ten states");
    exportStates("second_new",*secondNewest.snapshot);
    require(backend.window().valid(error),"Single-capacity window must preserve managed graph invariants");
}
void seedRetirementAndFacade() {
    Fixture f; BackendOptions options; options.windowSize=2; options.maxSolverTimeSeconds=3.0;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    for(uint64_t id=1;id<=3;++id) { append(backend,f,id); poseConstraints(backend,StateId(id)); }
    require(backend.addLandmark(LandmarkId(1),{1,0,0},std::nullopt,error),"uninitialized anchor");
    require(backend.addObservation(LandmarkId(1),{3,1,0},error),"uninitialized track continuation");
    auto result=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(result.accepted && backend.window().landmarks().find(LandmarkId(1))
        && backend.window().landmarks().find(LandmarkId(1))->anchor.frameId==3
        && result.marginalization->movedSeeds==1,"unabsorbed seed keeps its ID and surviving observation");
    require(!backend.window().landmarks().consumed({3,1,0}),"unconsumed seed measurements are not falsely marked historical");
    FixedLagBackend redundant(options); initialize(redundant,f);
    for(uint64_t id=1;id<=3;++id) { append(redundant,f,id); poseConstraints(redundant,StateId(id)); }
    require(redundant.setKeyframe(StateId(2),false)
        && redundant.addLandmark(LandmarkId(1),{2,0,0},std::nullopt,error)
        && redundant.addObservation(LandmarkId(1),{3,1,0},error),"seed starts on second-newest non-keyframe");
    const auto indexed=redundant.window().landmarks().inCamera(LandmarkId(1),StateId(3),1);
    require(indexed && indexed->key==KeypointIdentifier(3,1,0),"sorted track index resolves the occupied camera/epoch");
    const auto savedKeys=redundant.window().landmarks().ofLandmark(LandmarkId(1));
    require(!redundant.addObservation(LandmarkId(1),{3,1,1},error)
        && redundant.window().landmarks().ofLandmark(LandmarkId(1))==savedKeys
        && redundant.window().valid(error),"duplicate camera insertion rolls both contiguous indexes back atomically");
    const auto second=redundant.step(MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(second.accepted && second.marginalization->movedSeeds==1
        && redundant.window().landmarks().find(LandmarkId(1))->anchor.frameId==3
        && !redundant.window().graph().blocks.count({BlockKind::Landmark,1})
        && !redundant.window().landmarks().consumed({3,1,0}),
        "SECOND_NEW removes only the old seed view, preserving ID and fresh support without rho or prior reuse");
    require(redundant.window().landmarks().inFrame(StateId(2)).empty()
        && redundant.window().landmarks().inCamera(LandmarkId(1),StateId(3),1),"bulk frame erasure preserves the surviving reverse index");
    ViSlamBackend facade(options);
    require(facade.initializeRig(f.rig,kinematics::Transformation(),error),"facade rig");
    auto frame=f.frame(1); require(facade.addStates(*frame,f.prediction(1),{},error),"facade owned frame");
    require(facade.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"facade landmark");
    require(facade.addObservation(LandmarkId(1),StateId(1),1,0,error),"facade observation");
    require(facade.numFrames()==1 && facade.numLandmarks()==1 && facade.isObserved({1,1,0}),"single source of window membership");
    require(facade.multiFrame(StateId(1))->landmarkId(1,0)==1,"project-compatible detached frame view");
    MapPoints points; require(facade.getLandmarks(points)==1 && points.at(LandmarkId(1)).isInitialised,"project-compatible map view");
}
void priorCoupledRetirement() {
    class ExternalAnchorCost final : public ::ceres::SizedCostFunction<1,7,1> {
        bool Evaluate(double const* const* x,double* r,double** J) const override {
            r[0]=x[0][0]+x[1][0];
            if(J && J[0]) { std::fill(J[0],J[0]+7,0.0); J[0][0]=1; }
            if(J && J[1]) J[1][0]=1;
            return true;
        }
    };
    Fixture f; BackendOptions options; options.windowSize=2; options.maxSolverTimeSeconds=3.0;
    options.autoReanchorLandmarks=true; options.preserveOldestVisualInformation=false;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    for(uint64_t id=1;id<=3;++id) { append(backend,f,id); poseConstraints(backend,StateId(id)); }
    require(backend.addLandmark(LandmarkId(1),{2,0,0},f.rho(2,0),error),"surviving anchor");
    require(backend.addObservation(LandmarkId(1),{1,1,0},error),"old observation creates rho prior coupling");
    // Managed outgoing image observations are now deliberately dropped. Use a
    // real external incident cost to exercise the conservative coupled-prior path.
    FactorRecord external; external.keys={BlockKey::pose(StateId(1)),{BlockKind::Landmark,1}};
    external.cost=std::make_shared<ExternalAnchorCost>();
    require(backend.addFactor(std::move(external)).accepted(),"external rho/old-state coupling");
    auto first=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(first.accepted && backend.window().graph().prior,"historical prior created");
    const auto& keys=backend.window().graph().prior->keys();
    require(std::find(keys.begin(),keys.end(),BlockKey{BlockKind::Landmark,1})!=keys.end(),"rho is referenced by prior");
    append(backend,f,4); poseConstraints(backend,StateId(4));
    require(backend.addObservation(LandmarkId(1),{4,1,0},error),"new live measurement");
    auto retired=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(retired.accepted && retired.marginalization->reanchoredLandmarks.empty()
        && retired.marginalization->retiredLandmarks.size()==1,"prior-coupled outgoing point is not silently reparameterized");
    require(retired.marginalization->retirements.front().reason=="historical_prior_coupled",
        "prior-coupled retirement reason must be distinguishable from absent support");
    require(backend.window().landmarks().consumed({4,1,0}) && backend.window().valid(error),"retirement keeps historical-evidence and domain invariants");
}
void perStepBudgetAndHistory() {
    Fixture f; BackendOptions options; options.windowSize=3;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    append(backend,f,1); poseConstraints(backend,StateId(1));
    SolveBudget invalid; invalid.maxIterations=0;
    const auto rejected=backend.step(std::nullopt,{}, {}, {}, {},invalid);
    require(!rejected.accepted && !backend.snapshot() && !rejected.retiredState
        && backend.window().valid(error),"invalid per-step budget cannot publish or corrupt staged input");
    SolveBudget valid; valid.maxIterations=3; valid.minIterations=1; valid.timeSeconds=3.0; valid.numThreads=1;
    const auto accepted=backend.step(std::nullopt,{}, {}, {}, {},valid);
    require(accepted.accepted && accepted.snapshot->states.size()==1 && !accepted.retiredState,
        "per-step controls preserve normal accepted publication without eviction");
    // A deliberately slow first NIS evaluation must not consume Ceres' allowance.
    // Assert the solve was entered and accepted, never a machine-dependent time ceiling.
    struct DelayedFirstGateCost final : ::ceres::CostFunction {
        mutable std::atomic<bool> firstEvaluation{true};
        DelayedFirstGateCost() { set_num_residuals(1); mutable_parameter_block_sizes()->push_back(1); }
        bool Evaluate(double const* const* x,double* r,double** J) const override {
            if(firstEvaluation.exchange(false)) std::this_thread::sleep_for(std::chrono::milliseconds(60));
            r[0]=x[0][0];
            if(J && J[0]) J[0][0]=1;
            return true;
        }
    };
    BackendGraph delayed;
    const BlockKey key{BlockKind::Auxiliary,42};
    delayed.blocks.emplace(key,ParameterBlock::euclidean(key,Eigen::VectorXd::Zero(1)));
    FactorRecord factor; factor.id=1; factor.keys={key}; factor.lifecycle=FactorLifecycle::Active;
    factor.cost=std::make_shared<DelayedFirstGateCost>(); delayed.factors.emplace(factor.id,std::move(factor));
    SolveBudget ceresOnly; ceresOnly.timeSeconds=0.02; ceresOnly.minIterations=0;
    const auto afterSlowGate=BackendOptimizer(options).optimize(delayed,{},ceresOnly);
    require(afterSlowGate.accepted && afterSlowGate.passes==1 && afterSlowGate.ceresTimeLimitMs==20.0,
        "slow pre-gate cannot exhaust the Ceres-only time budget before the first solve");
}
void parallelNisGateAgreementAndRollback() {
    BackendGraph input; const BlockKey key{BlockKind::Auxiliary,9};
    input.blocks.emplace(key,ParameterBlock::euclidean(key,Eigen::VectorXd::Constant(1,0.2)));
    for(uint64_t i=1;i<=256;++i) {
        auto reference=input.blocks.at(key); if(i%3==0) reference.value[0]=0;
        auto factor=makeParameterPrior(reference,Eigen::MatrixXd::Identity(1,1));
        factor.id=i; factor.lifecycle=FactorLifecycle::Active; factor.nisThreshold=0.01;
        input.factors.emplace(i,std::move(factor));
    }
    BackendOptions settings; settings.maxIterations=4; settings.maxSolverTimeSeconds=10;
    settings.linearSolver=BackendLinearSolver::DenseQr;
    auto serial=input,parallel=input; SolveBudget budget; budget.numThreads=1;
    const auto first=BackendOptimizer(settings).optimize(serial,{},budget);
    budget.numThreads=4; const auto second=BackendOptimizer(settings).optimize(parallel,{},budget);
    require(first.accepted && second.accepted && first.preRejected==85 && second.preRejected==85,"Serial/parallel NIS decisions differ");
    require((serial.blocks.at(key).value-parallel.blocks.at(key).value).norm()<1e-12,"Parallel gating changed numeric solution");
    for(const auto& item:serial.factors) require(item.second.lifecycle==parallel.factors.at(item.first).lifecycle,"Parallel gating changed factor lifecycle");
    class InvalidCost final : public ::ceres::SizedCostFunction<1,1> {
    public:
        explicit InvalidCost(bool throwing):throwing_(throwing) {}
        bool Evaluate(double const* const*,double* r,double**) const override {
            if(throwing_) throw std::runtime_error("gate test exception");
            r[0]=std::numeric_limits<double>::quiet_NaN(); return true;
        }
    private: bool throwing_;
    };
    for(bool throwing:{false,true}) {
        auto invalid=input; auto bad=input.factors.begin()->second; bad.id=257;
        bad.cost=std::make_shared<InvalidCost>(throwing); invalid.factors.emplace(bad.id,bad);
        const auto rejected=BackendOptimizer(settings).optimize(invalid,{},budget);
        require(!rejected.accepted && invalid.blocks.at(key).value[0]==0.2,"Invalid parallel residual must restore graph values");
        for(const auto& item:invalid.factors) require(item.second.lifecycle==FactorLifecycle::Active,"Invalid parallel residual must restore all lifecycle changes");
    }
}
void factorProfilingPreservesEvaluation() {
    auto reference=ParameterBlock::euclidean({BlockKind::Auxiliary,1},Eigen::VectorXd::Ones(1));
    const auto original=makeParameterPrior(reference,2*Eigen::MatrixXd::Identity(1,1));
    ::cgmli::backend::detail::EvaluationCounters counters; ::cgmli::backend::detail::ProfiledCost cost(original.cost.get(),counters);
    double value=1.4; const double* parameters[]={&value}; double r=0,expected=0,J=0,expectedJ=0;
    double* request[]={&J}; double* expectedRequest[]={&expectedJ}; double* noDerivative[]={nullptr};
    require(cost.Evaluate(parameters,&r,request) && original.cost->Evaluate(parameters,&expected,expectedRequest),"profiled Jacobian evaluation");
    require(r==expected && J==expectedJ,"Factor profiling must preserve residual/Jacobian values");
    require(cost.Evaluate(parameters,&r,nullptr) && cost.Evaluate(parameters,&r,noDerivative),"profiled residual-only evaluation");
    const auto counts=counters.totals();
    require(counts.jacobianCalls==1 && counts.residualCalls==2,"Factor profiler must classify null Jacobian arrays correctly");
}
void realWindowResidualStatisticsContract() {
    class ConstantCost final : public ::ceres::SizedCostFunction<2,1> {
    public:
        ConstantCost(double a,double b):a_(a),b_(b) {}
        bool Evaluate(double const* const*,double* r,double** J) const override {
            r[0]=a_; r[1]=b_; if(J && J[0]) { J[0][0]=0; J[0][1]=0; } return true;
        }
    private: double a_,b_;
    };
    BackendGraph graph; const BlockKey key{BlockKind::Auxiliary,1};
    graph.blocks.emplace(key,ParameterBlock::euclidean(key,Eigen::VectorXd::Zero(1)));
    const auto add=[&](FactorId id,FactorKind kind,double a,double b,double threshold,bool robust) {
        FactorRecord f; f.id=id; f.kind=kind; f.keys={key}; f.lifecycle=FactorLifecycle::Active;
        f.cost=std::make_shared<ConstantCost>(a,b); f.nisThreshold=threshold;
        if(robust) f.loss=std::make_shared<::ceres::HuberLoss>(1.0);
        graph.factors.emplace(id,std::move(f));
    };
    add(1,FactorKind::Visual,1,2,9.21,true); add(2,FactorKind::Visual,10,0,9.21,true);
    add(3,FactorKind::Imu,3,0,std::numeric_limits<double>::infinity(),false);
    graph.prior=std::make_shared<MarginalizationPrior>(std::vector<ParameterBlock>{graph.blocks.at(key)},
        Eigen::MatrixXd::Identity(1,1),Eigen::VectorXd::Constant(1,0.4));
    BackendOptions settings; settings.maxIterations=10; settings.maxSolverTimeSeconds=30;
    const auto result=BackendOptimizer(settings).optimize(graph);
    require(result.accepted && result.residualStatisticsValid,"residual statistics do not change optimization admission");
    const auto& initial=result.preGateResiduals[0]; const auto& admitted=result.preSolveResiduals[0];
    const auto& final=result.postSolveResiduals[0];
    require(initial.factors==2 && initial.overGate==1 && admitted.factors==1 && final.factors==1,
        "pre-gate, admitted and fitted statistics must describe their own factor sets");
    debug::recordError("visual_pre_gate_nis_sum_error",std::abs(initial.squaredNorm-105),1e-12);
    debug::recordError("visual_admitted_nis_sum_error",std::abs(admitted.squaredNorm-5),1e-12);
    debug::recordError("visual_robust_cost_error",std::abs(admitted.robustCost-(std::sqrt(5.0)-0.5)),1e-12);
    debug::recordError("imu_nis_sum_error",std::abs(result.postSolveResiduals[1].squaredNorm-9),1e-12);
    require(result.postSolveResiduals[3].factors==1 && result.postSolveResiduals[3].components==1,
        "historical prior must be included exactly once and never robustified again");
    debug::recordError("optimized_prior_residual",result.postSolveResiduals[3].squaredNorm,1e-10);
    auto csv=debug::artifactFile("residual_statistics.csv");
    if(csv) {
        csv<<"phase,kind,factors,components,nis_sum,white_component_rms,robust_cost,nis_p95,over_gate\n";
        const ResidualStatisticsArray* phases[]{&result.preGateResiduals,&result.preSolveResiduals,&result.postSolveResiduals};
        for(int p=0;p<3;++p) for(int k=0;k<5;++k) {
            const auto& s=(*phases[p])[k]; csv<<p<<','<<k<<','<<s.factors<<','<<s.components<<','<<s.squaredNorm<<','<<s.componentRms()
                <<','<<s.robustCost<<','<<s.p95Nis<<','<<s.overGate<<'\n';
        }
    }
}
void refitRecoveryAndSchurSafety() {
    const auto insert=[](BackendGraph& graph,FactorRecord f) {
        f.id=graph.factors.size()+1; f.lifecycle=FactorLifecycle::Active; graph.factors.emplace(f.id,std::move(f));
    };
    BackendGraph input; const BlockKey x{BlockKind::Auxiliary,1};
    input.blocks.emplace(x,ParameterBlock::euclidean(x,Eigen::VectorXd::Zero(1)));
    auto target=input.blocks.at(x); target.value[0]=1;
    insert(input,makeParameterPrior(target,100*Eigen::MatrixXd::Identity(1,1)));
    auto weak=makeParameterPrior(input.blocks.at(x),Eigen::MatrixXd::Identity(1,1)); weak.nisThreshold=0.25; insert(input,weak);
    BackendOptions settings; settings.maxIterations=20; settings.maxSolverTimeSeconds=10; settings.maxOutlierRefits=0;
    auto limited=input; const auto bounded=BackendOptimizer(settings).optimize(limited);
    require(bounded.accepted && bounded.termination==SolveTermination::RefitBudget && bounded.unrefittedOutliers==1
        && limited.blocks.at(x).value[0]>0.99 && limited.factors.at(2).lifecycle==FactorLifecycle::Rejected,
        "Refit limit must preserve a usable state and remove live outliers instead of rolling back");
    require(bounded.postSolveResiduals[4].factors==1 && bounded.postSolveResiduals[4].overGate==0,
        "accepted limited statistics describe the cleaned factor set");
    auto postOnly=input; SolveBudget healthyTracking; healthyTracking.skipPreNis=true;
    const auto postOnlyResult=BackendOptimizer(settings).optimize(postOnly,{},healthyTracking);
    require(postOnlyResult.accepted && postOnlyResult.preNisSkipped && postOnlyResult.preNisMs==0
        && postOnlyResult.preGateResiduals[4].factors==0 && postOnlyResult.postRejected==1
        && postOnly.factors.at(2).lifecycle==FactorLifecycle::Rejected,
        "post-only gating still removes rejected evidence before marginalization");
    auto failed=input;
    const SolutionValidator rejectQuality=[](const BackendGraph&,std::string& reason) { reason="test quality failure"; return false; };
    const auto rejected=BackendOptimizer(settings).optimize(failed,rejectQuality);
    require(!rejected.accepted && rejected.termination==SolveTermination::Failure && failed.blocks.at(x).value[0]==0
        && failed.factors.at(2).lifecycle==FactorLifecycle::Active,
        "An actual quality failure at a refit limit must still restore values and factor lifecycle");
    auto oneIteration=settings; oneIteration.maxIterations=1; oneIteration.maxOutlierRefits=3;
    auto iterationLimited=input; const auto iterationResult=BackendOptimizer(oneIteration).optimize(iterationLimited);
    require(iterationResult.accepted && iterationResult.termination==SolveTermination::IterationBudget
        && iterationResult.unrefittedOutliers==1 && iterationLimited.blocks.at(x).value[0]>0.9,
        "iteration exhaustion is separate from estimation failure");
    class SlowFirstJacobian final : public ::ceres::SizedCostFunction<1,1> {
        mutable std::atomic<bool> firstJacobian{true};
        bool Evaluate(double const* const* values,double* residuals,double** jacobians) const override {
            if(jacobians && jacobians[0]) {
                if(firstJacobian.exchange(false)) std::this_thread::sleep_for(std::chrono::milliseconds(60));
                jacobians[0][0]=100;
            }
            residuals[0]=100*(values[0][0]-1); return true;
        }
    };
    auto timed=input; timed.factors.at(1).cost=std::make_shared<SlowFirstJacobian>();
    SolveBudget timeBudget; timeBudget.timeSeconds=0.02; timeBudget.minIterations=1;
    const auto timedResult=BackendOptimizer(settings).optimize(timed,{},timeBudget);
    require(timedResult.accepted && timedResult.ceresTimeBudgetReached && timedResult.termination==SolveTermination::TimeBudget
        && timedResult.unrefittedOutliers==1 && timed.blocks.at(x).value[0]>0.9
        && timed.factors.at(2).lifecycle==FactorLifecycle::Rejected,
        "Ceres time exhaustion keeps its last usable step after quality checks and live outlier cleanup");
    settings.maxOutlierRefits=3; auto recovered=input;
    const auto accepted=BackendOptimizer(settings).optimize(recovered);
    if(accepted.accepted) debug::recordError("known_scalar_solution_error",std::abs(recovered.blocks.at(x).value[0]-1),1e-6);
    require(accepted.problemBuilds==1 && accepted.problemReuses==accepted.passes-1,
        "outlier refits must reuse one problem without counting rejected costs twice");
    require(accepted.accepted && accepted.passes>=2 && accepted.iterations<=20 && accepted.postRejected==1
        && recovered.factors.at(2).lifecycle==FactorLifecycle::Rejected,"Internal refit must recover within the shared budget");

    class LastLandmarkCost final : public ::ceres::SizedCostFunction<1,1,1> {
        bool Evaluate(double const* const* x,double* r,double** J) const override {
            r[0]=x[0][0]; if(J && J[0]) J[0][0]=1; if(J && J[1]) J[1][0]=0; return true;
        }
    };
    auto orphan=input; const BlockKey unused{BlockKind::Landmark,9};
    orphan.blocks.emplace(unused,ParameterBlock::euclidean(unused,Eigen::VectorXd::Constant(1,0.5)));
    FactorRecord last; last.cost=std::make_shared<LastLandmarkCost>(); last.keys={x,unused};
    last.nisThreshold=0.25; insert(orphan,last);
    const auto cleaned=BackendOptimizer(settings).optimize(orphan);
    require(cleaned.accepted && cleaned.problemBuilds==1 && cleaned.problemReuses>0
        && cleaned.activeLandmarks==0 && cleaned.linearSolverUsed!="SPARSE_SCHUR",
        "removing the last landmark residual must drop its Ceres block and empty Schur group before refit");

    BackendGraph independent;
    independent.blocks.emplace(x,ParameterBlock::euclidean(x,Eigen::VectorXd::Zero(1)));
    insert(independent,makeParameterPrior(independent.blocks.at(x),Eigen::MatrixXd::Identity(1,1)));
    std::vector<ParameterBlock> references;
    for(uint64_t id:{1u,2u}) {
        const BlockKey key{BlockKind::Landmark,id}; auto block=ParameterBlock::euclidean(key,Eigen::VectorXd::Constant(1,0.5));
        independent.blocks.emplace(key,block); references.push_back(block);
        insert(independent,makeParameterPrior(block,Eigen::MatrixXd::Identity(1,1)));
    }
    auto separated=independent; const auto separable=BackendOptimizer(settings).optimize(separated);
    require(separable.accepted && separable.schurIndependent && separable.activeLandmarks==2,"Independent landmarks must be recognized");
    auto denseSettings=settings; denseSettings.linearSolver=BackendLinearSolver::DenseSchur;
    auto denseGraph=independent; const auto dense=BackendOptimizer(denseSettings).optimize(denseGraph);
    require(dense.accepted && dense.linearSolverUsed=="DENSE_SCHUR","explicit dense Schur uses independent landmarks");
    for(const auto& block:separated.blocks) debug::recordError("dense_schur_solution_"+std::to_string(block.first.id),
        (block.second.value-denseGraph.blocks.at(block.first).value).norm(),1e-8);
    Eigen::MatrixXd A(1,2); A<<1,1;
    independent.prior=std::make_shared<MarginalizationPrior>(references,A,Eigen::VectorXd::Zero(1));
    const auto coupled=BackendOptimizer(settings).optimize(independent);
    require(coupled.accepted && !coupled.schurIndependent && coupled.linearSolverUsed!="SPARSE_SCHUR"
        && coupled.priorColumns==2,"A prior coupling landmarks must prevent Schur elimination");
    const auto denseCoupled=BackendOptimizer(denseSettings).optimize(independent);
    require(denseCoupled.accepted && denseCoupled.linearSolverUsed!="DENSE_SCHUR",
        "dense Schur also preserves a coupled prior through the general solver fallback");
}
void budgetLimitedSolvePreservesCleanPrior() {
    class PoseBridge final : public ::ceres::SizedCostFunction<1,7,7> {
        bool Evaluate(double const* const* p,double* r,double** J) const override {
            r[0]=p[1][0]-p[0][0]-0.1;
            if(J && J[0]) { std::fill(J[0],J[0]+7,0.0); J[0][0]=-1; }
            if(J && J[1]) { std::fill(J[1],J[1]+7,0.0); J[1][0]=1; }
            return true;
        }
    };
    class IncidentOutlier final : public ::ceres::SizedCostFunction<1,1,7> {
        bool Evaluate(double const* const* p,double* r,double** J) const override {
            r[0]=p[0][0];
            if(J && J[0]) J[0][0]=1;
            if(J && J[1]) std::fill(J[1],J[1]+7,0.0);
            return true;
        }
    };
    Fixture fixture; BackendOptions options; options.windowSize=2; options.maxIterations=20;
    options.maxSolverTimeSeconds=10; options.maxOutlierRefits=0;
    FixedLagBackend backend(options); initialize(backend,fixture); std::string error;
    for(uint64_t id=1;id<=3;++id) { append(backend,fixture,id); poseConstraints(backend,StateId(id)); }
    FactorRecord bridge; bridge.cost=std::make_shared<PoseBridge>();
    bridge.keys={BlockKey::pose(StateId(1)),BlockKey::pose(StateId(2))}; bridge.epochs={StateId(1),StateId(2)};
    require(backend.addFactor(bridge).accepted(),"pose bridge produces a nonempty surviving prior");
    const BlockKey auxiliary{BlockKind::Auxiliary,77};
    auto parameter=ParameterBlock::euclidean(auxiliary,Eigen::VectorXd::Zero(1));
    require(backend.addParameter(parameter,error),"limited prior test auxiliary parameter");
    auto target=parameter; target.value[0]=1;
    require(backend.addFactor(makeParameterPrior(target,100*Eigen::MatrixXd::Identity(1,1))).accepted(),"limited prior test target");
    FactorRecord outlier; outlier.cost=std::make_shared<IncidentOutlier>();
    outlier.keys={auxiliary,BlockKey::pose(StateId(1))}; outlier.epochs={StateId(1)}; outlier.nisThreshold=0.25;
    const auto submission=backend.addFactor(outlier);
    require(submission.accepted(),"incident outlier is initially below its gate");
    const auto result=backend.step(MarginalizationPolicy::MARGIN_OLD);
    require(result.accepted && result.solve.termination==SolveTermination::RefitBudget
        && result.solve.unrefittedOutliers==1 && result.snapshot && result.marginalization && result.marginalization->success,
        "a quality-checked limited solve must publish and marginalize normally");
    const auto& graph=backend.window().graph();
    require(graph.prior && !graph.factors.count(submission.id) && backend.window().valid(error),
        "rejected live cost is archived before marginalization");
    require(std::find(graph.prior->keys().begin(),graph.prior->keys().end(),auxiliary)==graph.prior->keys().end(),
        "the rejected incident cost must not create auxiliary information in the historical prior");
    debug::recordError("limited_solve_retained_scalar_error",std::abs(graph.blocks.at(auxiliary).value[0]-1),0.001);
}
void frontendBatchAtomicity() {
    Fixture f; BackendOptions options; options.windowSize=3;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    const auto before=backend.window().estimate(StateId(2));
    const kinematics::Transformation seed(Eigen::Vector3d(0.11,0,0),Eigen::Quaterniond::Identity());
    VisualUpdateBatch batch;
    batch.landmarks.push_back({LandmarkId(1),{1,0,0},f.rho(1,0),{}});
    batch.observations.push_back({LandmarkId(1),{2,1,0},{}});
    batch.observations.push_back({LandmarkId(1),{2,99,0},{}});
    require(!backend.applyFrontendBatch(StateId(2),batch,seed,SensorHealth::Degraded,false,error),"late invalid view rejects complete frontend batch");
    require(backend.window().landmarks().size()==0 && backend.window().nextLandmarkId()==LandmarkId(1),"rejected batch keeps associations and ID high-water mark");
    const auto after=backend.window().estimate(StateId(2));
    require((before->T_WB.T()-after->T_WB.T()).norm()==0 && before->visual==after->visual
        && before->keyframe==after->keyframe,"pose, health and keyframe rollback with tracks");
    batch.observations.pop_back();
    require(backend.applyFrontendBatch(StateId(2),batch,seed,SensorHealth::Degraded,false,error),"frontend batch commit");
    require(backend.window().landmarks().ofLandmark(LandmarkId(1)).size()==2
        && backend.window().estimate(StateId(2))->visual==SensorHealth::Degraded
        && backend.window().estimate(StateId(2))->lidar==before->lidar
        && !backend.window().estimate(StateId(2))->keyframe && !backend.snapshot(),"batch stages exactly one anchor plus one view, preserves lidar, no publication");
    VisualUpdateBatch historical;
    historical.observations.push_back({LandmarkId(1),{1,1,0},{}});
    require(!backend.applyFrontendBatch(StateId(2),historical,std::nullopt,SensorHealth::Valid,true,error),"current transaction cannot insert historical extra observations");
    require(!backend.applyFrontendBatch(StateId(1),{},seed,SensorHealth::Valid,true,error),"cannot reopen historical epoch");
    VisualUpdateBatch discardMature; discardMature.retiredLandmarks.push_back(LandmarkId(1));
    require(!backend.applyFrontendBatch(StateId(2),discardMature,std::nullopt,SensorHealth::Valid,false,error)
        && backend.window().landmarks().find(LandmarkId(1))->initialized,"frontend capacity cleanup must never retire an optimized point");
    require(backend.window().valid(error),"domain valid after batch rejection");
}
void frontendConsensusTransaction() {
    Fixture f; BackendOptions options; options.windowSize=3; options.maxSolverTimeSeconds=3.0;
    FixedLagBackend backend(options); initialize(backend,f); std::string error;
    append(backend,f,1); append(backend,f,2);
    require(backend.addLandmark(LandmarkId(1),{1,0,0},f.rho(1,0),error),"frontend transaction old landmark");
    require(backend.addObservation(LandmarkId(1),{1,1,0},error),"frontend transaction old stereo depth");
    require(backend.addObservation(LandmarkId(1),{2,1,0},error),"frontend transaction current observation");
    require(backend.addLandmark(LandmarkId(2),{2,0,1},f.rho(2,0,1),error),"frontend transaction geometric anchor");
    require(backend.addObservation(LandmarkId(2),{2,1,1},error),"frontend transaction new stereo depth");
    const auto before=backend.window().estimate(StateId(2))->T_WB;
    const kinematics::Transformation seed(Eigen::Vector3d(0.12,0.01,0),Eigen::Quaterniond::Identity());
    require(!backend.applyFrontendUpdate(StateId(2),seed,{{2,1,0},{2,0,1}},error),"anchor rejection aborts complete consensus transaction");
    require(backend.window().landmarks().find(KeypointIdentifier{2,1,0})
        && (backend.window().estimate(StateId(2))->T_WB.T()-before.T()).norm()==0,"failed pruning rolls back previously removed observation and pose");
    require(backend.applyFrontendUpdate(StateId(2),seed,{{2,1,0}},error),"newest unpublished pose and pruning commit together");
    require(!backend.window().landmarks().find(KeypointIdentifier{2,1,0})
        && (backend.window().estimate(StateId(2))->T_WB.T()-seed.T()).norm()<1e-12,"consensus committed to authoritative graph");
    require(!backend.applyFrontendUpdate(StateId(1),seed,{},error),"historical epoch cannot be reseeded");
    poseConstraints(backend,StateId(1)); poseConstraints(backend,StateId(2));
    const auto result=backend.step(); require(result.accepted,"publish initialized state");
    require(!backend.applyFrontendUpdate(StateId(2),before,{},error),"published pose cannot be overwritten by frontend initialization");
    require(!backend.applyFrontendBatch(StateId(2),{},std::nullopt,SensorHealth::Valid,true,error),"published epoch cannot receive another frontend batch");
    require(backend.snapshot()==result.snapshot && backend.window().valid(error),"rejected reseed leaves published snapshot and window intact");
}
}
#ifdef CGMLI_EMBED_MODULE_TESTS
void cgmli::debug::registerBackendCases(std::vector<cgmli::debug::ModuleCase>& cases) {
    cases.push_back({"jacobian","factorGeometryAndJacobians",[]() { factorGeometryAndJacobians(); }});
    cases.push_back({"jacobian","analyticSphereDerivativeAgreement",[]() { analyticSphereDerivativeAgreement(); }});
    cases.push_back({"backend","cachedVisualGateMatchesReference",[]() { cachedVisualGateMatchesReference(); }});
    cases.push_back({"jacobian","visualPoseCacheMatchesUncachedTrials",[]() { visualPoseCacheMatchesUncachedTrials(); }});
    cases.push_back({"backend","ownershipAndIndices",[]() { ownershipAndIndices(); }});
    cases.push_back({"backend","perStepBudgetAndHistory",[]() { perStepBudgetAndHistory(); }});
    cases.push_back({"backend","parallelNisGateAgreementAndRollback",[]() { parallelNisGateAgreementAndRollback(); }});
    cases.push_back({"backend","factorProfilingPreservesEvaluation",[]() { factorProfilingPreservesEvaluation(); }});
    cases.push_back({"backend","realWindowResidualStatisticsContract",[]() { realWindowResidualStatisticsContract(); }});
    cases.push_back({"backend","constraintGradientBalanceIsRobustAndReadOnly",[]() { constraintGradientBalanceIsRobustAndReadOnly(); }});
    cases.push_back({"backend","rigRotationSolvePreservesTranslationAndPriorCoordinates",[]() { rigRotationSolvePreservesTranslationAndPriorCoordinates(); }});
    cases.push_back({"backend","refitRecoveryAndSchurSafety",[]() { refitRecoveryAndSchurSafety(); }});
    cases.push_back({"marginalization","budgetLimitedSolvePreservesCleanPrior",[]() { budgetLimitedSolvePreservesCleanPrior(); }});
    cases.push_back({"backend","frontendBatchAtomicity",[]() { frontendBatchAtomicity(); }});
    cases.push_back({"backend","frontendConsensusTransaction",[]() { frontendConsensusTransaction(); }});
    cases.push_back({"marginalization","windowPoliciesAndRollback",[]() { windowPoliciesAndRollback(); }});
    cases.push_back({"marginalization","tenFrameWindowUsesOneInsertionSlot",[]() { tenFrameWindowUsesOneInsertionSlot(); }});
    cases.push_back({"marginalization","seedRetirementAndFacade",[]() { seedRetirementAndFacade(); }});
    cases.push_back({"marginalization","priorCoupledRetirement",[]() { priorCoupledRetirement(); }});
    cases.push_back({"marginalization","marginalizationQuadraticAgreement",[]() { marginalizationQuadraticAgreement(); }});
    cases.push_back({"marginalization","parallelNormalEquationsPreservePrior",[]() { parallelNormalEquationsPreservePrior(); }});
    cases.push_back({"marginalization","untouchedPriorAndDiagonalKernel",[]() { untouchedPriorAndDiagonalKernel(); }});
    cases.push_back({"marginalization","redundantFrameKeepsTracksSparse",[]() { redundantFrameKeepsTracksSparse(); }});
    cases.push_back({"marginalization","oldestFrameRetainsPriorFreeTracks",[]() { oldestFrameRetainsPriorFreeTracks(); }});
    cases.push_back({"marginalization","oldestSchurKeepsIdentityWithoutReusingEvidence",[]() { oldestSchurKeepsIdentityWithoutReusingEvidence(); }});
    cases.push_back({"marginalization","reanchorInformationAudit",[]() { reanchorInformationAudit(); }});
    cases.push_back({"marginalization","reanchorFixedBearingChangeIsRejected",[]() { reanchorFixedBearingChangeIsRejected(); }});
    cases.push_back({"marginalization","reanchorFullCoordinateTransportReference",[]() { reanchorFullCoordinateTransportReference(); }});
}
#else
int main() {
    try {
        ownershipAndIndices(); factorGeometryAndJacobians(); analyticSphereDerivativeAgreement(); windowPoliciesAndRollback(); tenFrameWindowUsesOneInsertionSlot(); seedRetirementAndFacade(); priorCoupledRetirement(); perStepBudgetAndHistory(); parallelNisGateAgreementAndRollback(); factorProfilingPreservesEvaluation(); refitRecoveryAndSchurSafety(); budgetLimitedSolvePreservesCleanPrior(); frontendBatchAtomicity(); frontendConsensusTransaction();
        marginalizationQuadraticAgreement(); reanchorInformationAudit(); reanchorFixedBearingChangeIsRejected(); reanchorFullCoordinateTransportReference(); realWindowResidualStatisticsContract();
        parallelNormalEquationsPreservePrior();
        untouchedPriorAndDiagonalKernel(); redundantFrameKeepsTracksSparse();
        oldestFrameRetainsPriorFreeTracks();
        oldestSchurKeepsIdentityWithoutReusingEvidence(); cachedVisualGateMatchesReference(); visualPoseCacheMatchesUncachedTrials();
        constraintGradientBalanceIsRobustAndReadOnly();
        rigRotationSolvePreservesTranslationAndPriorCoordinates();
        std::cout<<"managed backend regression passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

#endif // CGMLI_EMBED_MODULE_TESTS
