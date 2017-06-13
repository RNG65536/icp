#include <iostream>
using std::cout;
using std::endl;
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen\Dense>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
typedef glm::dvec2 GlmVec2;
typedef glm::dvec3 GlmVec3;
typedef glm::dvec4 GlmVec4;
typedef glm::dquat GlmQuat;
typedef glm::dmat3 GlmMat3;
typedef glm::dmat4 GlmMat4;
typedef MatrixXd EigenMatrix;
typedef VectorXd EigenVector;

void verify()
{
    EigenMatrix C = EigenMatrix::Identity(6, 6);
    C(2, 3) = 0.5;
    C(3, 2) = 0.5;

    EigenVector b(6);
    b << 1, 2, 3, 4, 5, 6;
    cout << C << endl;
    cout << b << endl;

    EigenVector x0 = C.fullPivHouseholderQr().solve(b);
    EigenVector x1 = C.fullPivLu().solve(b);
    EigenVector x2 = C.jacobiSvd(ComputeFullU | ComputeFullV).solve(b);
    EigenVector x3 = C.llt().solve(b); // symmetric, p.d.
    EigenVector x4 = C.ldlt().solve(b); // symmetric, p.s.d.
    cout << x0 << endl;
    cout << x1 << endl;
    cout << x2 << endl;
    cout << x3 << endl;
    cout << x4 << endl;
}

EigenVector LUsolve(const EigenMatrix& C, const EigenMatrix& b)
{
    return C.fullPivLu().solve(b);
}
