#pragma once

#include "WindowVisualFactor.hpp"
#include <ceres/problem.h>

namespace cgmli::backend::detail {

struct VisualAuditReport {
    bool host=false,cuda=false,normal=false,schur=false;
    bool direct=false;
    double residual=0,jacobian=0,normalH=0,normalG=0,reducedH=0,reducedG=0;
    double directH=0,directG=0,directStep=0;
};

VisualAuditReport auditVisualWindow(::ceres::Problem&,VisualEvaluationCache&,
    const std::vector<double*>& variables,const std::vector<int>& retained,const std::vector<int>& eliminated,
    const std::vector<::ceres::ResidualBlockId>& visual,double huberScale);

}
