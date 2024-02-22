#include <ceres/ceres.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>

// 定义刚体变换残差结构体
// 刚体变换的轴角表示残差结构体
struct RigidBodyTransformResidual {
    RigidBodyTransformResidual(const Eigen::Vector3d& source, const Eigen::Vector3d& target)
        : source_point(source), target_point(target) {}

    template <typename T>
    bool operator()(const T* const rotation_params, const T* const translation, T* residuals) const {
        // 将轴角参数分配给Eigen类型
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> axis_angle(rotation_params);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation_vector(translation);

        // 构造旋转矩阵
        Eigen::Matrix<T, 3, 3> rotation_matrix = Eigen::AngleAxis<T>(axis_angle.norm(), axis_angle.normalized()).toRotationMatrix();

        // 刚体变换公式：transformed_point = R * source_point + translation
        Eigen::Matrix<T, 3, 1> transformed_point = rotation_matrix * source_point.cast<T>() + translation_vector;

        // 计算残差：目标点与变换后的源点之间的差异
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals);
        residuals_map = transformed_point - target_point.cast<T>();

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& source, const Eigen::Vector3d& target) {
        return new ceres::AutoDiffCostFunction<RigidBodyTransformResidual, 3, 3, 3>(
            new RigidBodyTransformResidual(source, target)
        );
    }

private:
    Eigen::Vector3d source_point;  // 本体坐标系中的点
    Eigen::Vector3d target_point;  // 全局坐标系中的点
};


// 定义轴角参数化的 Local Parameterization
class AxisAngleParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // x_plus_delta = x * exp(delta)
        Eigen::Map<const Eigen::Vector3d> x_map(x);
        Eigen::Map<const Eigen::Vector3d> delta_map(delta);
        Eigen::Map<Eigen::Vector3d> x_plus_delta_map(x_plus_delta);


        Eigen::Quaterniond x_quaternion(Eigen::AngleAxisd(x_map.norm(), x_map.normalized()));
        Eigen::Quaterniond delta_quaternion(Eigen::AngleAxisd(delta_map.norm(), delta_map.normalized()));
        
        Eigen::Quaterniond x_plus_quaternion = x_quaternion * delta_quaternion ;

        // 将四元数转换为轴角
        Eigen::AngleAxisd axis_angle(x_plus_quaternion);
        x_plus_delta_map = axis_angle.axis() * axis_angle.angle();// result_quaternion.coeffs().segment(1, 3);

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        // 计算 Jacobian 矩阵
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_map(jacobian);
        jacobian_map.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 3; }
    virtual int LocalSize() const { return 3; }
};

int main() {
    // 创建优化问题
    ceres::Problem problem;

    // 添加参数块（旋转和平移）
    double rotation[3] = {0.01, 0.0, 0.0};  // 初始旋转
    double translation[3] = {1.0, 2.0, 5.0};  // 初始平移
    problem.AddParameterBlock(rotation, 3, new AxisAngleParameterization);
    problem.AddParameterBlock(translation, 3);

    // 添加多个点对的残差项（刚体变换）
    std::vector<Eigen::Vector3d> model_points{
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 1.0)
    };
    std::vector<Eigen::Vector3d> measured_points;
    Eigen::Vector3d axis_angle(1.0 * M_PI/ 3.0, 1.0 * M_PI/ 3.0, 0.0);
    Eigen::Vector3d translation_v(1.0, 0.0, 0.0);
    Eigen::Matrix3d rotation_matrix = Eigen::AngleAxisd(axis_angle.norm(), axis_angle.normalized()).toRotationMatrix();
    for(size_t i = 0; i < model_points.size(); i++){
        Eigen::Vector3d transformed_point = rotation_matrix * model_points[i] + translation_v;
        measured_points.push_back(transformed_point);
        std::cout<<"["<<transformed_point.transpose()<<"]"<<std::endl;
    }


    for (size_t i = 0; i < model_points.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<RigidBodyTransformResidual, 3, 3, 3>(
                new RigidBodyTransformResidual(model_points[i], measured_points[i])
            );

        problem.AddResidualBlock(cost_function, nullptr, rotation, translation);
    }

    // 配置优化选项
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.minimizer_progress_to_stdout = true;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Optimized Rotation: " << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << std::endl;
    std::cout << "Optimized Translation: " << translation[0] << ", " << translation[1] << ", " << translation[2] << std::endl;

    return 0;
}
