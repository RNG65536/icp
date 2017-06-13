// Iterative Closest Point for Simultaneous Localization and Mapping
// see http://blog.csdn.net/fuxingyin/article/details/51425721

#include <iostream>
using std::cout;
using std::endl;
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen\Dense>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
#ifndef NDEBUG
#define DEBUGMODE
#define NDEBUG
#endif
#include <GL/freeglut.h>
#ifdef DEBUGMODE
#undef NDEBUG
#undef DEBUGMODE
#endif
#define M_PI 3.14159265358979323846
#include <random>
std::default_random_engine rng;
std::uniform_real_distribution<double> dist(0.0, 1.0);
double randf()
{
    return dist(rng);
}
typedef glm::dvec2 GlmVec2;
typedef glm::dvec3 GlmVec3;
typedef glm::dvec4 GlmVec4;
typedef glm::dquat GlmQuat;
typedef glm::dmat3 GlmMat3;
typedef glm::dmat4 GlmMat4;
typedef MatrixXd EigenMatrix;
typedef VectorXd EigenVector;

bool g_auto_solve = true;

void verify();
EigenVector LUsolve(const EigenMatrix& C, const EigenMatrix& b);

std::ostream& operator<<(std::ostream& out, const GlmVec3& v)
{
    out << v.x << ", " << v.y << ", " << v.z;
    return out;
}

struct Point
{
    GlmVec3 pos;
    GlmVec3 nl;

    Point(){}
    Point(const GlmVec3& _pos, const GlmVec3& _nl) : pos(_pos), nl(_nl) {}
    void draw(double c)
    {
        glColor3f(1, 1, 1);
        glVertex3dv(value_ptr(pos));
        glColor3f(c, 1 - c, 0);
        glVertex3dv(value_ptr(pos + nl * 0.1));
    }
};

// typedef int HashId;
// struct HashCell
// {
//     std::vector<Point*> points;
//     HashCell(const GlmVec3& c, const GlmVec3& p)
//     {
// 
//     }
// };
// class HashGrid
// {
//     std::unordered_map<HashId, HashCell*> _map;
// public:
//     HashCell *hash(const GlmVec3& p)
//     {
//         HashId id = 0;
//         double px = ceil(p.x / 5.0);
//         double py = ceil(p.y / 5.0);
//         double pz = ceil(p.z / 5.0);
//         int x = (int)px;
//         int y = (int)py;
//         int z = (int)pz;
//         id = x + (y << 10) + (z << 20);
//         if (_map.find(id) == _map.end())
//         {
//             _map[id] = new HashCell(GlmVec3(x, y, z), GlmVec3(px, py, pz) * 5.0);
//         }
//         auto pr = _map[id];
//         assert(pr);
//         return pr;
//     }
//     ~HashGrid()
//     {
//         for (auto& h : _map)
//         {
//             delete h.second;
//         }
//     }
// };

int count = 1000;
std::vector<Point> dst;
std::vector<Point> src;

GlmVec3 r0 = GlmVec3(0.1, 0.05, 0.03) * 1.0; // must not be large
GlmVec3 t0 = GlmVec3(0.01, 0.02, 0.03) * 5.0; // can be large
EigenVector x = EigenVector::Zero(6); // approximated (r, t)

double E()
{
    GlmVec3 r = GlmVec3(x(0), x(1), x(2));
    GlmVec3 t = GlmVec3(x(3), x(4), x(5));
    cout << "r = " << r << endl;
    cout << "t = " << t << endl;

    double e = 0;
    for (int i = 0; i < dst.size(); i++)
    {
        GlmVec3& p = src[i].pos;
        GlmVec3& q = dst[i].pos;
        GlmVec3& n = dst[i].nl;
        double s = dot(p - q, n) + dot(r, cross(p, n)) + dot(t, n);
        e += s * s;
    }

    return e;
}

void solve()
{
    // 1) find correspondence (nearest neighbor)
    //      for debugging set src2dst[i] = i
    std::vector<int> src2dst(src.size());
    for (int i = 0; i < src.size(); i++)
    {
        double min_dist = 1e10;
        for (int j = 0; j < dst.size(); j++)
        {
            double d = length(dst[j].pos - src[i].pos);
            if (d < min_dist)
            {
                min_dist = d;
                src2dst[i] = j;
            }
        }
    }

    // 2) solve linear constraint
    EigenMatrix C = EigenMatrix::Zero(6, 6);
    EigenMatrix b = EigenMatrix::Zero(6, 1);
    for (int i = 0; i < src.size(); i++)
    {
        GlmVec3& p = src[i].pos;
//         GlmVec3& q = dst[i].pos;
//         GlmVec3& n = dst[i].nl;
        GlmVec3& q = dst[src2dst[i]].pos;
        GlmVec3& n = dst[src2dst[i]].nl;
        GlmVec3 pxn = cross(p, n);
//         cout << pxn << endl;
        EigenMatrix w(6, 1);
        w << pxn.x, pxn.y, pxn.z, n.x, n.y, n.z;
        C += w * w.transpose();

        double pqn = dot(p - q, n);
        b += w * pqn;
    }

    cout << endl;
    cout << C << endl;
    cout << endl;
    cout << b << endl;
    cout << endl;
    x = LUsolve(C, -b);
    cout << x << endl;
    cout << endl;

    // must calculate this before actually applying the (r, t) transform, since the transform is inlined in E
    cout << "E = " << E() << endl;

    // apply transform
    // Rp + t approx. p + rxp + t
    GlmVec3 r = GlmVec3(x(0), x(1), x(2));
    GlmVec3 t = GlmVec3(x(3), x(4), x(5));
    for (auto& p : src)
    {
        p.pos += cross(r, p.pos) + t;
    }
}

void init()
{
    dst.clear();
    src.clear();
    for (int i = 0; i < count; i++)
    {
        double rnd1 = randf();
        double rnd2 = randf();
        double cos_theta = 1 - rnd1 + rnd1 * 0.8;
        double sin_theta = sqrt(1 - cos_theta * cos_theta);
        double phi = rnd2 * M_PI * 2;
        GlmVec3 p = GlmVec3(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi)) * (1.0 + 0.1 * randf());
//         dst.push_back(Point(p, normalize(p)));
        dst.push_back(Point(p, GlmVec3(randf() - 0.5, randf() - 0.5, randf() - 0.5)));

//         dst.push_back(Point(GlmVec3(randf(), randf(), randf()),
//             normalize(GlmVec3(randf() - 0.5, randf() - 0.5, randf() - 0.5))));
    }

    GlmMat4 rot = mat4_cast(normalize(GlmQuat(r0)));
    //     vec3 rr = eulerAngles(quat_cast(rot));
    for (auto& p : dst)
    {
        src.push_back(Point(
            GlmVec3(rot * GlmVec4(p.pos, 1)) + t0
            + GlmVec3(randf(), randf(), randf()) * 0.01, // add some noise, which is typical in scanning
            GlmMat3(rot) * p.nl * -1.0));
    }

    cout << "E = " << E() << endl;
}

void keyboard(unsigned char key, int x, int y)
{
    if (key == 'q')
    {
        exit(0);
    }
    else if (key == ' ')
    {
        g_auto_solve = !g_auto_solve;
//         solve();
    }
    else if (key == 'r')
    {
        init();
    }
}

double rot_y = 0;
void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glRotatef(rot_y, 0, 1, 0);
    rot_y += 1;

    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);
    glColor3f(0, 1, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1, 0);
    glColor3f(0, 0, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 1);
    glEnd();

    glBegin(GL_LINES);
    for (auto& p : dst)
    {
        p.draw(0);
    }
    for (auto& p : src)
    {
        p.draw(1);
    }
    glEnd();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glutSwapBuffers();
    glutPostRedisplay();

    if (g_auto_solve)
    {
        static int idx = 0;
        if (0 == ++idx % 10)
        {
            solve();
        }
    }
}

int main(int argc, char **argv)
{
    init();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(600, 600);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("");

    glPointSize(8);
    glEnable(GL_POINT_SMOOTH);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, 1, 0.1, 100);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(3, 2, 1, 0, 0, 0, 0, 1, 0);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMainLoop();

    return 0;
}








