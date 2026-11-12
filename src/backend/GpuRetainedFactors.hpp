#pragma once

#include <vector>

namespace cgmli::backend::detail {

struct FrozenImuData {
    double duration=0,gravity=0;
    double sensorPosition[3]{},sensorQuaternion[4]{},gyros[6]{};
    double deltaQuaternion[4]{},deltaVelocity[3]{},deltaPosition[3]{},referenceBias[6]{};
    double rotationGyro[9]{},velocityGyro[9]{},positionGyro[9]{},velocityAccel[9]{},positionAccel[9]{};
    double whitening[225]{};
};

enum class GpuChart { Euclidean, LeftSE3, RotationOnly };
struct GpuRetainedBlock {
    int offset=0, column=0, width=0, size=0;
    GpuChart chart=GpuChart::Euclidean;
};
struct GpuVariable {
    GpuRetainedBlock state;
    int lift=0;
};
struct GpuImuFactor {
    GpuRetainedBlock blocks[4];
    FrozenImuData imu;
    int firstRow=0;
};
struct GpuPriorBlock {
    GpuRetainedBlock state;
    int reference=0,chartOffset=0,tangentSize=0;
};
struct GpuPriorFactor {
    int firstBlock=0,blockCount=0,firstRow=0,rows=0,columns=0;
    int matrix=0,residual=0,chartOffset=0;
};
struct GpuRetainedFactors {
    std::vector<GpuImuFactor> imu;
    std::vector<GpuPriorFactor> priors;
    std::vector<GpuPriorBlock> priorBlocks;
    std::vector<double> priorData; ///< References, square-root matrices (row-major), and residuals.
    int rows=0,chartSize=0;
};

}
