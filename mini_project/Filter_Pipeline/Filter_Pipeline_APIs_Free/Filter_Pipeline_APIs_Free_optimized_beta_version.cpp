//-- Kushagra Shukla , Greeshwar R S
//-- 3D Perception using PointNet on FPGA
//-- EYSIP 2020-21 
// This script is API free version of file Filter_Pipeline_APIs.cpp so kindly refer it if you want to experiment using framework PCL 


// Importing basic Header files 
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono> // To meausre execution time of different algorthims
#define io ios_base::sync_with_stdio(false);cin.tie(NULL); // For fast I/O stream
using namespace std::chrono;
using namespace std;

// These are used for clustering purpose uisng DBSCAN Method
constexpr auto UNCLASSIFIED = -1;
constexpr auto CORE_POINT = 1;
constexpr auto BORDER_POINT = 2;
constexpr auto NOISE = -2;
constexpr auto SUCCESS = 0;
constexpr auto FAILURE = -3;


// These are used for optimization part 
#pragma optimization_level 3
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx")
#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")
#pragma GCC optimization ("unroll-loops")

class point_and_box //This class has been created to store the individual points in our cloud with idx and box information
{
public:
    int idx;
    int box;
    point_and_box(int arg_idx) {
        idx = arg_idx;  // points index in the input cloud
        box = -1;   // to indicate which box it belongs
    }
    // for comparing user defined class
    bool operator < (const point_and_box& rhs) const
    {
        return(box < rhs.box);
    }

};

template <class T> class PointXYZ //Defining our own Template class for the points in our 3D cloud
{
public:
    T x, y, z;
    int clusterID = -1; //Assinging every point with cluster ID as -1 which will change after DBSCAN asssigns them a new ID 

    PointXYZ() {    }

    PointXYZ(T arg_x, T arg_y, T arg_z) {
        x = arg_x;
        y = arg_y;
        z = arg_z;
    }
    // for equating 2 points
    PointXYZ <T>& operator = (const PointXYZ <T>& rhs) {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

    PointXYZ(const PointXYZ <T>& rhs) {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
    }
    // for adding 2 points 
    PointXYZ <T>& operator+=(const PointXYZ <T>& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
};

class Downsampling //Class to Downsample the cloud and hence reduce the computation on further pipeline
{

public:
    void getMinMax(vector< PointXYZ <long double> >& inCloud, PointXYZ <long double>& minp, PointXYZ <long double>& maxp) //Helper Function to find min/max from cloud
    {
        for (int i = 0; i < inCloud.size(); i++)
        {
            //min
            minp.x = min(minp.x, inCloud[i].x);
            minp.y = min(minp.y, inCloud[i].y);
            minp.z = min(minp.z, inCloud[i].z);
            //max
            maxp.x = max(maxp.x, inCloud[i].x);
            maxp.y = max(maxp.y, inCloud[i].y);
            maxp.z = max(maxp.z, inCloud[i].z);
        }
    }


    void voxel_downsample(PointXYZ <long double>& voxel_size, vector< PointXYZ <long double> >& inCloud, vector< PointXYZ <long double> >& outCloud, vector< point_and_box >& indices)
    {
        // min value of a point cloud is stored here
        PointXYZ <long double> minp(DBL_MAX, DBL_MAX, DBL_MAX);
        // max value of a point cloud is stored here
        PointXYZ <long double> maxp(-DBL_MAX, -DBL_MAX, -DBL_MAX);
        getMinMax(inCloud, minp, maxp);


        PointXYZ <long double> leafSize;
        leafSize.x = voxel_size.x;
        leafSize.y = voxel_size.y;
        leafSize.z = voxel_size.z;

        PointXYZ <long double> inv_leafSize(1.0 / leafSize.x, 1.0 / leafSize.y, 1.0 / leafSize.z);

        //Compute the minimum and maximum bounding box values
        PointXYZ <int> minb(static_cast<int> (floor(minp.x * inv_leafSize.x)),
            static_cast<int> (floor(minp.y * inv_leafSize.y)),
            static_cast<int> (floor(minp.z * inv_leafSize.z)));

        PointXYZ <int> maxb(static_cast<int> (floor(maxp.x * inv_leafSize.x)),
            static_cast<int> (floor(maxp.y * inv_leafSize.y)),
            static_cast<int> (floor(maxp.z * inv_leafSize.z)));

        PointXYZ <int> divb(maxb.x - minb.x + 1, maxb.y - minb.y + 1, maxb.z - minb.z + 1);
        PointXYZ <int> divb_mul(1, divb.x, divb.x * divb.y);

        //Go over all points and insert them into the right leaf
        for (int i = 0; i < inCloud.size(); i++) {
            int ijk0 = static_cast<int> (floor(inCloud[i].x * inv_leafSize.x) - minb.x);
            int ijk1 = static_cast<int> (floor(inCloud[i].y * inv_leafSize.y) - minb.y);
            int ijk2 = static_cast<int> (floor(inCloud[i].z * inv_leafSize.z) - minb.z);
            int idx = ijk0 * divb_mul.x + ijk1 * divb_mul.y + ijk2 * divb_mul.z;
            indices[i].box = idx;//the box index the point it belongs to
        }


        sort(indices.begin(), indices.end(), less<point_and_box>()); // sorting the indices

        //for calculating centroid for each voxel
        for (int cp = 0; cp < inCloud.size();)
        {
            PointXYZ <long double> centroid(inCloud[indices[cp].idx].x, inCloud[indices[cp].idx].y, inCloud[indices[cp].idx].z);
            int i = cp + 1;
            while (i < inCloud.size() && indices[cp].box == indices[i].box) {
                centroid += inCloud[indices[i].idx];
                ++i;
            }
            centroid.x /= static_cast<long double>(i - cp);
            centroid.y /= static_cast<long double>(i - cp);
            centroid.z /= static_cast<long double>(i - cp);

            outCloud.push_back(centroid);
            cp = i;
        }
    }

};

class PlaneSegmentation //Class to detect the plane using RANSAC method
{
public:
    vector< PointXYZ <long double> > inlier_cloud_result; // to store inlier points
    vector< PointXYZ <long double> > outlier_cloud_result; // to store inlier points
    double coeff_arr_result[4]{};


    PlaneSegmentation() {

    }
    void Plane_RANSAC_found(vector< PointXYZ <long double> >& cloud, double plane_coeff[4], const float distanceTol)
    {
        // for finding and returning the inliers corresponding to the given plane equation provided 
        // Measure distance between every point and fitted plane
        // If distance is smaller than threshold count(distanceTol) it as inlier

        // Ax + By + Cz - d = 0   -- storing the coeff of the plane
        const double a = plane_coeff[0];
        const double b = plane_coeff[1];
        const double c = plane_coeff[2];
        const double d = plane_coeff[3];

        for (int i = 0; i < cloud.size(); i++)
        {
            PointXYZ<long double>point = cloud[i];
            double x4 = point.x;
            double y4 = point.y;
            double z4 = point.z;

            // Estimate the distance of each point and check if it is wichin distanceTol
            const double D = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);
            if (D < distanceTol)
            {
                this->inlier_cloud_result.push_back(cloud[i]); // pushing inlier points
            }
            else
            {
                this->outlier_cloud_result.push_back(cloud[i]); // pushing outlier points
            }
        }

    }

    void Plane_RANSAC_find(vector< PointXYZ <long double> >& cloud, const int maxIterations, const float distanceTol)
    {
        srand(time(NULL));
        const int points2fit = 3;
        for (int i = 0; i < maxIterations; i++)
        {
            // 1. Select three randon points

            vector< PointXYZ <long double> > inlier_cloud; // to store inlier points
            vector< PointXYZ <long double> > outlier_cloud; // to store outlier points

            double coeff_arr[4]{};   // to store the plane co-efficient

            for (int i = 0; i < points2fit; i++)
            {
                inlier_cloud.push_back(cloud[(rand() % cloud.size())]); // storing the randomly generated points
            }

            // Get random points form the cloud
            long double x1, y1, z1;
            long double x2, y2, z2;
            long double x3, y3, z3;

            // storing the 3 randomly found points
            x1 = inlier_cloud[0].x;
            y1 = inlier_cloud[0].y;
            z1 = inlier_cloud[0].z;

            x2 = inlier_cloud[1].x;
            y2 = inlier_cloud[1].y;
            z2 = inlier_cloud[1].z;

            x3 = inlier_cloud[2].x;
            y3 = inlier_cloud[2].y;
            z3 = inlier_cloud[2].z;

            // calculate
            // 2. Calculate the plane that these points form
            // Ax + By + Cz - d = 0
            const double a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
            const double b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
            const double c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
            const double d = -(a * x1 + b * y1 + c * z1);
            // storing plane coefficient
            coeff_arr[0] = a;
            coeff_arr[1] = b;
            coeff_arr[2] = c;
            coeff_arr[3] = d;

            // Measure distance between every point and fitted plane
            // If distance is smaller than threshold count it as inlier
            for (int i = 0; i < cloud.size(); i++)
            {
                PointXYZ<long double>point = cloud[i];
                long double x4 = point.x;
                long double y4 = point.y;
                long double z4 = point.z;

                // 3. Estimate the distance of each point and check if it is wichin distanceTol
                const double D = fabs(a * x4 + b * y4 + c * z4 + d) / sqrt(a * a + b * b + c * c);
                if (D < distanceTol)
                {
                    inlier_cloud.push_back(cloud[i]); // pushing inlier points
                }
                else
                {
                    outlier_cloud.push_back(cloud[i]); // pushing outlier points
                }
            }

            // the fit with the most inliers will be selected
            if (inlier_cloud.size() > inlier_cloud_result.size())
            {
                //inliersResult = inliers;
                this->coeff_arr_result[0] = a;
                this->coeff_arr_result[1] = b;
                this->coeff_arr_result[2] = c;
                this->coeff_arr_result[3] = d;
                inlier_cloud_result = inlier_cloud;
                outlier_cloud_result = outlier_cloud;
            }
        }

    }
};


struct Node // Structure to represent node of kd tree 
{
    PointXYZ<long double> point;
    int id;
    Node* left;
    Node* right;

    Node(PointXYZ<long double> pt, int setId) : point(pt), id(setId), left(NULL), right(NULL) {}
};

struct KdTree //Structure created to implement kdtree data structure for the point cloud considering x,y and z axises
{

    Node* root;

    KdTree()
        : root(nullptr) {}

    void insertHelper(Node** node, int depth, PointXYZ<long double>& point, int id)
    {
        if (*node == nullptr)
        {
            (*node) = new Node(point, id);
        }
        else
        {
            int cd = depth % 3;  // 3 dim kd-tree

            if (cd == 0)
            {
                if (point.x < (*node)->point.x)
                    insertHelper(&(*node)->left, depth + 1, point, id);
                else
                    insertHelper(&(*node)->right, depth + 1, point, id);
            }
            else if (cd == 1)
            {
                if (point.y < (*node)->point.y)
                    insertHelper(&(*node)->left, depth + 1, point, id);
                else
                    insertHelper(&(*node)->right, depth + 1, point, id);
            }
            else
            {
                if (point.z < (*node)->point.z)
                    insertHelper(&(*node)->left, depth + 1, point, id);
                else
                    insertHelper(&(*node)->right, depth + 1, point, id);
            }
        }
    }

    void insert(PointXYZ<long double>& point, int id)
    {
        insertHelper(&root, 0, point, id);
    }

    void searchHelper(PointXYZ<long double>& pivot, Node* node, int depth, float distanceTol, std::vector<int>& ids)
    {
        if (node != NULL) {
            if (((node->point.x >= (pivot.x - distanceTol)) && (node->point.x <= (pivot.x + distanceTol))) && ((node->point.y >= (pivot.y - distanceTol)) && (node->point.y <= (pivot.y + distanceTol))))
            {
                float distance = sqrt((node->point.x - pivot.x) * (node->point.x - pivot.x) + (node->point.y - pivot.y) * (node->point.y - pivot.y) + (node->point.z - pivot.z) * (node->point.z - pivot.z));

                if (distance <= distanceTol) {
                    ids.push_back(node->id);
                }
            }

            int cd = depth % 3;  // 3 dim kd-tree

            if (cd == 0)
            {
                if ((pivot.x - distanceTol) < node->point.x) {
                    searchHelper(pivot, node->left, depth + 1, distanceTol, ids);
                }

                if ((pivot.x + distanceTol) > node->point.x) {
                    searchHelper(pivot, node->right, depth + 1, distanceTol, ids);
                }
            }
            else if (cd == 1) {
                if ((pivot.y - distanceTol) < node->point.y) {
                    searchHelper(pivot, node->left, depth + 1, distanceTol, ids);
                }
                if ((pivot.y + distanceTol) > node->point.y) {
                    searchHelper(pivot, node->right, depth + 1, distanceTol, ids);
                }
            }
            else
            {
                if ((pivot.z - distanceTol) < node->point.z) {
                    searchHelper(pivot, node->left, depth + 1, distanceTol, ids);
                }
                if ((pivot.z + distanceTol) > node->point.z) {
                    searchHelper(pivot, node->right, depth + 1, distanceTol, ids);
                }
            }

        }
    }

    // return a list of point ids in the tree that are within distance of pivot
    std::vector<int> search(PointXYZ <long double>& pivot, float distanceTol)
    {
        std::vector<int> ids;
        searchHelper(pivot, root, 0, distanceTol, ids);

        return ids;
    }
};

// this is a kd tree constructed only using 2 coordinates x and y 
struct KdTree_2d {

    Node* root;

    KdTree_2d()
        : root(nullptr) {}

    void insertHelper(Node** node, int depth, PointXYZ<long double> point, int id) {
        if (*node == nullptr) {
            (*node) = new Node(point, id);
        }
        else {
            int cd = depth % 2;  // 2 dim kd-tree

            if (cd == 0) {
                if (point.x < (*node)->point.x)
                    insertHelper(&(*node)->left, depth + 1, point, id);
                else
                    insertHelper(&(*node)->right, depth + 1, point, id);
            }
            else {
                if (point.y < (*node)->point.y)
                    insertHelper(&(*node)->left, depth + 1, point, id);
                else
                    insertHelper(&(*node)->right, depth + 1, point, id);
            }
        }
    }

    void insert(PointXYZ<long double> point, int id) {
        insertHelper(&root, 0, point, id);
    }

    void searchHelper(PointXYZ<long double> pivot, Node* node, int depth, float distanceTol, std::vector<int>& ids) {
        if (node != NULL) {
            //cout << "yayay1" << endl;
            if ((node->point.x >= (pivot.x - distanceTol) && (node->point.x <= (pivot.x + distanceTol))) && (node->point.y >= (pivot.y - distanceTol) && (node->point.y <= (pivot.y + distanceTol))))
            {
                float distance = sqrt((node->point.x - pivot.x) * (node->point.x - pivot.x) + (node->point.y - pivot.y) * (node->point.y - pivot.y));

                if (distance <= distanceTol) {
                    ids.push_back(node->id);
                }
            }
            //cout << "yayay2" << endl;
            if (depth % 2 == 0) // 2 dim kd-tree
            {
                if ((pivot.x - distanceTol) < node->point.x) {
                    searchHelper(pivot, node->left, depth + 1, distanceTol, ids);
                }

                if ((pivot.x + distanceTol) > node->point.x) {
                    searchHelper(pivot, node->right, depth + 1, distanceTol, ids);
                }
            }
            else {
                if ((pivot.y - distanceTol) < node->point.y) {
                    searchHelper(pivot, node->left, depth + 1, distanceTol, ids);
                }
                if ((pivot.y + distanceTol) > node->point.y) {
                    searchHelper(pivot, node->right, depth + 1, distanceTol, ids);
                }
            }
            // cout << "yayay3" << endl;

        }
    }

    // return a list of point ids in the tree that are within distance of pivot
    std::vector<int> search(PointXYZ <long double> pivot, float distanceTol) {
        std::vector<int> ids;
        searchHelper(pivot, root, 0, distanceTol, ids);

        return ids;
    }
};




class DBSCAN //This class help in clustering the ROI objects finally at the end of the pipeline
{
public:
    DBSCAN(unsigned int minPts, float eps, vector< PointXYZ <long double> >& points, KdTree* t) //Constructor
    {
        m_minPoints = minPts; //Minimum point to decide whther a point is core or not
        m_epsilon = eps; // Square of sphercial radious 
        m_points = points; // This basically represents our 3D point cloud only
        m_pointSize = points.size(); // 3D Point cloud size
        tree = t; //KD tree for serching neighbors

    }
    ~DBSCAN() {}

    int run()
    {
        int clusterID = 1; // Assign ID to clusters
        total_clusters = 1; // Counts total number of clusters
        vector< PointXYZ <long double> >::iterator iter;
        for (iter = m_points.begin(); iter != m_points.end(); ++iter) //Start iterating the point in a cloud
        {

            if (iter->clusterID == UNCLASSIFIED) //If point is yet not traversed
            {
                if (expandCluster(*iter, clusterID) != FAILURE) //If we ultimately get success in clustering we have to assign a new cluster and incrase the size of cluster too
                {
                    total_clusters += 1;
                    clusterID += 1;
                }
            }
        }

        return 0;
    }
    vector<int> calculateCluster(PointXYZ <long double>& point)
    {
        std::vector<int> clusterIndex = tree->search(point, pow(m_epsilon, 1.0 / 2.0));
        return clusterIndex;
    }

    int expandCluster(PointXYZ <long double>& point, int clusterID)
    {
        vector<int> clusterSeeds = calculateCluster(point); //Calculates all possible neighbors within range of radious of sphere

        if (clusterSeeds.size() < m_minPoints) //Checking if pint is a noise than return failure
        {
            point.clusterID = NOISE;
            return FAILURE;
        }
        else // Case when point is core or border point 
        {
            int index = 0, indexCorePoint = 0;
            vector<int>::iterator iterSeeds;
            for (iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds) // Iterating the neighbouring points using iterSeeds
            {
                m_points.at(*iterSeeds).clusterID = clusterID; // For each neighbouring points assigning it as a new cluster ID thus putting it into classified category 
                if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y && m_points.at(*iterSeeds).z == point.z)
                {
                    indexCorePoint = index; // This steps assigns index number to indexCorePoint for the core point
                }
                ++index; //Incrementing the index by unity for each neighboring point
            }

            clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint); // This step will remove the location of core point from clusterSeeds vector

            // This loop will iterate neighbouring points and perform the same procedure as decribed above 

            for (vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i)
            {
                vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i])); //Calculates all possible neighbors within range of radious of sphere

                if (clusterNeighors.size() >= m_minPoints)
                {
                    vector<int>::iterator iterNeighors;
                    for (iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors)
                    {
                        if (m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE)
                        {
                            if (m_points.at(*iterNeighors).clusterID == UNCLASSIFIED)
                            {
                                clusterSeeds.push_back(*iterNeighors);
                                n = clusterSeeds.size();
                            }
                            m_points.at(*iterNeighors).clusterID = clusterID;

                        }
                    }
                }
            }

            return SUCCESS;
        }
    }
    inline double calculateDistance(const PointXYZ <long double>& pointCore, const PointXYZ <long double>& pointTarget)
    {
        return pow(pointCore.x - pointTarget.x, 2) + pow(pointCore.y - pointTarget.y, 2) + pow(pointCore.z - pointTarget.z, 2);
    }

    int getTotalPointSize() { return m_pointSize; }
    int getMinimumClusterSize() { return m_minPoints; }
    int getEpsilonSize() { return m_epsilon; }
public:
    vector< PointXYZ <long double> > m_points;
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
    int total_clusters = 0;
    KdTree* tree;
};


// [NOTE:] This section is under optimization and it doesn't have any role in our pipeline it was our intital idea to use Eucledian clustering but the code was failing
////////////////////// The following snippets are part of Eucledian clustering however this piece of code is yet to be optmizied and completed ////////

/*
Following issues were reported from above code and thus we decided to switch for DBSCAN algorithm finally.
1.  The code was failing for a large amount of point cloud size, due to deep recursion in the algorithm we were getting stack overflow, that was corrected by adding
    "#pragma comment(linker, "/STACK:200000000")" at the top of CPP script however still this issue persist and code lags a lot
2.	We found on executing code with some 3D scans we were not getting exact clusters, and we were unable to figure out the exact reason.
*/

/*
* To call the Eucledian clustering and test it from main() one may use the following
* EuclideanCluster(cloud_name for which u want to do clustering, kd-tree of the same cloud, 0.02 distance tolerance, 100 min no. points in each cluster, 2500 max no. of points);
*/

/*
void ClusterHelper(size_t i, vector< PointXYZ <long double> > cloud, std::vector<size_t>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol)
{
    processed[i] = true;
    cluster.push_back(i);
    std::vector<size_t> nearest = tree->search(cloud[i], distanceTol);
    for (size_t id : nearest)
    {
        if (!processed[id])
            ClusterHelper(id, cloud, cluster, processed, tree, distanceTol);
    }
}
void EuclideanCluster(vector< PointXYZ <long double> > cloud, KdTree* tree, float distanceTol, int minSize, int maxSize)
{
    vector<vector< PointXYZ <long double> > > clusters;
    vector<bool> processed(cloud.size(), false);
    for (size_t idx = 0; idx < cloud.size(); ++idx)
    {
        if (!processed[idx])
        {
            std::vector<size_t> cluster_idx;
            vector< PointXYZ <long double> > cluster;
            ClusterHelper(idx, cloud, cluster_idx, processed, tree, distanceTol);
            if (cluster_idx.size() >= minSize && cluster_idx.size() <= maxSize)
            {
                for (size_t i : cluster_idx)
                {
                    cluster.push_back(cloud[i]);
                }
                clusters.push_back(cluster);
                cout << cluster.size() << endl;
            }
            else
            {
                for (size_t i = 1; i < cluster_idx.size(); i++)
                {
                    processed[cluster_idx[i]] = false;
                }
            }
        }
    }
    cout << clusters.size();
    ofstream zc("C:\\Users\\1kusp\\OneDrive\\Documents\\Run_pro\\apifreeop\\see.pcd");
    zc << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n";
    zc << "WIDTH " << clusters[0].size() << endl;
    zc << "HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";
    zc << "POINTS " << clusters[0].size() << endl;
    zc << "DATA ascii\n";
    for (int i = 0; i < clusters[0].size(); i++)
    {
        zc << clusters[0][i].x << " " << clusters[0][i].y << " " << clusters[0][i].z << " " << 16711680 << endl;
    }
}
*/
bool plane_check(vector< PointXYZ <long double> > planeCloud, double a, double b, double c, double d) {
    if (planeCloud.size() < 2) { // 2 is just an assumption not proceed if cloud size less than 2
        return false;
    }


    long double max_x = planeCloud[0].x;
    long double min_x = planeCloud[0].x;
    long double max_y = planeCloud[0].y;
    long double min_y = planeCloud[0].y;


    auto* tree = new KdTree_2d;
    for (int i = 0; i < planeCloud.size(); i++) {
        tree->insert(planeCloud[i], i);
    }
    // finding the max and min in x, y, and z values of points
    for (int i = 1; i < planeCloud.size(); i++) {
        if (planeCloud[i].x > max_x) {
            max_x = planeCloud[i].x;
        }
        if (planeCloud[i].x < min_x) {
            min_x = planeCloud[i].x;
        }
        if (planeCloud[i].y > max_y) {
            max_y = planeCloud[i].y;
        }
        if (planeCloud[i].y < min_y) {
            min_y = planeCloud[i].y;
        }
    }

    // storing the extracted point cloud in z axis(as in int main) for visualization
    cout << "plane_check : " << min_x << " " << max_x << " " << min_y << " " << max_y << endl;

    ofstream y("D:\\plc_try1\\pcdss\\z_compare.pcd");

    y << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n";

    y << "WIDTH " << planeCloud.size() << endl;

    y << "HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";

    y << "POINTS " << planeCloud.size() << endl;

    y << "DATA ascii\n";


    for (int i = 0; i < planeCloud.size(); i++)
    {
        y << planeCloud[i].x << " " << planeCloud[i].y << " " << planeCloud[i].z << " " << 16711680 << endl;
    }
    y.close();

    // storing the constructed plane cloud to compare pcd files
    ofstream yp("D:\\plc_try1\\pcdss\\compare_cloud.pcd");
    vector< PointXYZ <long double> > compare_cloud;

    yp << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n";

    yp << "WIDTH " << compare_cloud.size() << endl;

    yp << "HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";

    yp << "POINTS " << compare_cloud.size() << endl;

    yp << "DATA ascii\n";



    /*
    The reason for this algorithm is to check if the extracted plane is actually a plane or not.(used only along z axis RANSAC)
    Main algorithm is - we will construct a plane parallel to the extracted plane point cloud(using RANSAC plane segmentation) and we will compare each point
    on the plane to the extracted plane point cloud. so if within a certain radius(search_radius) to that point more than min_neigh no. of points are present then that
    point on the constructed point cloud will be considered as occupied. This way we will search for all points. and finally based on occupied and unoccupied points 
    we will calculate the percentage of occupied points, so if this percentage is greater than a certain percent we will consider it to be a plane and remove the plane form
    original point cloud*/
    double occupied = 0;
    double unoccupied = 0;

    double grid_size = 0.005; // represents the distance between the points in the constructed point.
    double search_radius = 0.006; // radius to consider a point to be occupied or not.
    int min_neigh = 1; // minimum neighbours present within the search radius to consider the point is occupied or not
    for (long double i = min_y; i <= max_y; i += grid_size) {
        for (long double j = min_x; j <= max_x; j += grid_size) {
            PointXYZ <long double> pt;
            // points on the plane parallel to the extracted plane
            pt.x = j;
            pt.y = i;
            pt.z = -(a * pt.x + b * pt.y + d) / c;
            // comparing the points on the plane constructed to the extracted plane
            std::vector<int> clusterIndex = tree->search(pt, search_radius);
            if (clusterIndex.size() >= min_neigh) {
                occupied++;
            }
            else {
                unoccupied++;
            }
            yp << pt.x << " " << pt.y << " " << pt.z << " " << 16711680 << endl;
            compare_cloud.push_back(pt);
        }
    }

    yp.close();
    double percent_occupied = (occupied / (occupied + unoccupied)); // percentage occupied
    cout << "percentage occupied : " << percent_occupied << endl;

    if (percent_occupied >= 0.5) { // if this percentage is greater than  50% consider it to be a plane
        return true;
    }
    return false;

}
int main()
{
    io; // Fast I/O stream 
    auto start = high_resolution_clock::now(); //To meausre execution time 

    ifstream infile("D:\\plc_try1\\pcdss\\3bottle_light_2.pcd"); //To fetch the file from local disk
    string my; //To read PCD file line by line 

    long long int line = 0; //counts the line
    long long int total = 0; //stores the total line in the PCD file
    long long int color; //stores the RGB packed value
    vector<point_and_box> indices; //To store required indexes

    PointXYZ <long double> point(0.0, 0.0, 0.0); //TO store XYZ value
    vector< PointXYZ <long double> > inCloud, downCloud; // To store the points and clouds 

    auto passthrough_s = high_resolution_clock::now();


    while (getline(infile, my))
    {
        if (line == 9) // That line contains point information
        {

            int start = 0;
            string del = " ";
            int end = my.find(del);
            while (end != -1)
            {
                start = end + del.size();
                end = my.find(del, start);
            }
            total = stoi(my.substr(start, end - start));
        }

        if (line >= 11)
        {

            int start = 0;
            string del = " ";
            int end = my.find(del);
            vector <long double> doublek;
            while (end != -1)
            {
                doublek.push_back(stof(my.substr(start, end - start)));
                start = end + del.size();
                end = my.find(del, start);
            }
            color = stoi(my.substr(start, end - start));

            /*
             * Passthrough filtering and loading the points NOTE chk variable defines the range
             */

            float chk = 0.8; //Define the range of point cloud for pass through filter in (m)  //passthrough in x,y,z
            if ((doublek[0] < chk && doublek[0] > -chk) && (doublek[1] < chk && doublek[1] > -chk) && (doublek[2] < chk && doublek[2] > -chk))
            {
                indices.push_back(point_and_box(inCloud.size()));
                point.x = doublek[0];
                point.y = doublek[1];
                point.z = doublek[2];
                inCloud.push_back(point);
            }
        }
        line++;
    }
    infile.close(); // Closing the file

    cout << "TOTAL POINTS: " << total << "\nAFTER PASS THROUGH FILTER: " << inCloud.size() << endl;
    auto passthrough_e = high_resolution_clock::now();
    auto passthrough_d = duration_cast<microseconds>(passthrough_e - passthrough_s);
    cout << "Time taken by [Loading PCD file & Passthrough filter] is equal to (in seconds):::::::::::::  " << passthrough_d.count() / 1e+6 << endl;
    cout << "NOTE: Time taken by Passthrough filter is very small and large amount of time is taken for loading and storing PCD file\n";


    /*
    * Downsampling the points
    */

    auto downsample_s = high_resolution_clock::now();
    float vsize = 0.005;
    PointXYZ <long double>voxel_size(vsize, vsize, vsize);  // the voxel size (in X Y and Z axis) 
    Downsampling down_samp_pts;
    down_samp_pts.voxel_downsample(voxel_size, inCloud, downCloud, indices); // Calling class function for the downsampling of points 

    cout << "After Downsampling: " << downCloud.size() << endl;
    auto downsample_e = high_resolution_clock::now();
    auto downsample_d = duration_cast<microseconds>(downsample_e - downsample_s);
    cout << "Time taken by [Downsampling] is equal to (in seconds):::::::::::::  " << downsample_d.count() / 1e+6 << endl;
    vector< PointXYZ <long double> > cloud = downCloud; // Copy of downCloud

    // Saving the downsampled cloud (donwCloud)

    ofstream outfile("D:\\plc_try1\\pcdss\\downcloud.pcd");
    outfile << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n";
    outfile << "WIDTH " << downCloud.size() << endl;
    outfile << "HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";
    outfile << "POINTS " << downCloud.size() << endl;
    outfile << "DATA ascii\n";

    for (int i = 0; i < downCloud.size(); i++)
    {
        outfile << downCloud[i].x << " " << downCloud[i].y << " " << downCloud[i].z << " " << 16711680 << endl;
    }
    outfile.close();
  

    /*
    * Extraction of Planes using RANSAC
    */
    auto ransac_s = high_resolution_clock::now();
    const int maxIterations = 1000;
    const float distanceTol = 0.01;
    int chky = 1, chkz = 1;  // These are flags that turns-off if largest plane is extracted out along ground or in front of us respectively becz over that objects are supposed to be

    //Vectors that will basically push the clouds of extracted plane as well as point cloud lying below and above that plane 
    //for both ground and front planes, obviously if plane if important to us that is it may contains object
    std::vector< vector< PointXYZ <long double> >    >  planey;
    std::vector< vector< PointXYZ <long double> >    >  planez;
    std::vector< vector< PointXYZ <long double> >    >  cloudy;
    std::vector< vector< PointXYZ <long double> >    >  cloudz;
    int idy = 0, idz = 0; // Will count the planes wrt ground and front planes respectively
    long int nps = downCloud.size();  // It contains the total number of point cloud data

    float a{}, b{}, c{}, d{}; //Represents cofrecients of the plane 

    while ((cloud.size() > 0.01 * nps) && (cloud.size() != 1) && (chky || chkz)) //We are finding the planes until inliers have atleast 1% of the total size or if desired plane is not detected yet
    {
        //cout << cloud.size() << endl; 

        PlaneSegmentation plane_seg;
        plane_seg.Plane_RANSAC_find(cloud, maxIterations, distanceTol);// finding the plane
        // stroring plane coefficients
        a = plane_seg.coeff_arr_result[0];
        b = plane_seg.coeff_arr_result[1];
        c = plane_seg.coeff_arr_result[2];
        d = plane_seg.coeff_arr_result[3];

        vector< PointXYZ <long double> >    cloudpp = plane_seg.inlier_cloud_result; // Use to store the plane (after RANSAC)
        vector< PointXYZ <long double> >    cloudff = plane_seg.outlier_cloud_result; // Use to store the remaining cloud (after removing the plane)

        cloud.swap(cloudff);

        //red (x), green (y), and blue (z) this shows the axis color when we see cloud on visually
        double valz = acos(c / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))) * 180 / 3.1415; // in degrees (angle made with positive z axis)
        double valy = acos(b / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))) * 180 / 3.1415; // in degrees (angle made with positive z axis)
        double valx = acos(a / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))) * 180 / 3.1415; // in degrees (angle made with positive z axis)
        //cout << "Angle with  X Y Z  axis are :" << valx << "   " << valy << "    " << valz << endl;

        // lb and ub tells about tilt of plane in perfect case ground plane is supposed to align with y axis and same for plane in front of us 
        // but we are dealing with nosiy data and FOV of camera may capture the image at some angle hence we allow max 40 degree of tilt for both ground and vertical plane
        int lb = 40;
        int ub = 140;
        // Cehcking the angle if it is in range as described above
        bool fz = (valz > 0 && valz < lb) || (valz < 180 && valz > ub);
        bool fy = (valy > 0 && valy < lb) || (valy < 180 && valy > ub);
        bool fx = (valx > 0 && valx < lb) || (valx < 180 && valx > ub);

        if (fy && chky) // Ground Planes extraction 
        {
            chky = 0; // We turn this flag-off as we don't need further planes if you want you can turn it on but do the suitable changes by undertanding the full code

            //cout << " The above plane is useful along y axis\n";
            // Here we will extract the plane using coffceints from original cloud - downCloud
            PlaneSegmentation found_plane_seg_y;
            found_plane_seg_y.Plane_RANSAC_found(downCloud, plane_seg.coeff_arr_result, distanceTol);///plane_seg.coeff_arr_result ---- { a,b,c,d };

            vector< PointXYZ <long double> >    cloudppp = found_plane_seg_y.inlier_cloud_result; // Use to store the plane (after RANSAC)
            vector< PointXYZ <long double> >    cloudfff = found_plane_seg_y.outlier_cloud_result; // Use to store the remaining cloud (after removing the plane)

            downCloud = cloudfff;  // changing downcloud to the outliers (cloud after segmenting plane)
            idy++;
           

        }

        if (fz && chkz) // Front Planes extraction 
        {
            chkz = 0; // We turn this flag-off as we don't need further planes if you want you can turn it on but do the suitable changes by undertanding the full code
            // Here we will extract the plane using coffceints from original cloud - downCloud
            PlaneSegmentation found_plane_seg_z;
            found_plane_seg_z.Plane_RANSAC_found(downCloud, plane_seg.coeff_arr_result, distanceTol);///plane_seg.coeff_arr_result ---- { a,b,c,d };

            vector< PointXYZ <long double> >    cloudppp = found_plane_seg_z.inlier_cloud_result; // Use to store the plane (after RANSAC)
            vector< PointXYZ <long double> >    cloudfff = found_plane_seg_z.outlier_cloud_result; // Use to store the remaining cloud (after removing the plane)

            if (plane_check(cloudppp, a, b, c, d)) { // to check whether the obtained plane does not contain the ROI objects in the point cloud if not then we can remove that point
               
                cout << " The above plane is useful along z axis\n";
                downCloud = cloudfff; // changing downcloud to the outliers (cloud after segmenting plane)
                idz++;

            }

        }
    }

    //std::cerr << "Total Number of useful Planes present along ground = " << idy << std::endl;
    //std::cerr << "Total Number of useful Planes present in front of us = " << idz << std::endl;
    // You will always get idy and idz as [1 or 0] however if you have that flag (chky and chkz) always turned on you will see change to it.
    /*[IMP] It is assumed that you have only one plane in Y or Z hence the code is written in that way, you must do suitable changes if you are working on multiple planes*/
    auto ransac_e = high_resolution_clock::now();
    auto ransac_d = duration_cast<microseconds>(ransac_e - ransac_s);
    cout << "Time taken by [RANSAC] is equal to (in seconds):::::::::::::  " << ransac_d.count() / 1e+6 << endl;
    /*
    * Extraction of Objects using Clustering concept from the extracted ground or front planes
    */
    auto cls_s = high_resolution_clock::now();

        // Creation of fstream class object
        ofstream outfile_1;
        outfile_1.open("C:\\Users\\1kusp\\OneDrive\\Documents\\Run_pro\\apifreeop\\fcl.txt");
        int c_idx = 0; // It will count the total number of clusters
        // clustering and storing clusters into clusters vector
        std::vector<vector<PointXYZ<long double>>> clusters;
        //Defining parameters of clustering 
        float DISTANCE_TOL_CLUSTERING = 0.02; // radius of sphere in which we will search the MINIMUM_POINTS criteria to decide a point is core point or not
        int MIN_POINTS_IN_CLUSTER = 100, MAX_POINTS_IN_CLUSTER = 25000;
        double EPSILON = DISTANCE_TOL_CLUSTERING * DISTANCE_TOL_CLUSTERING;  // distance for clustering, metre^2
        int  MINIMUM_POINTS = 4;  // Minimum number of neighbors (data points) within epsilon radius.

        vector<PointXYZ<long double>> cloudf = downCloud; // final cloud after plane segmentation
        // Creating the KdTree object for the search method of the extraction
        auto* tree = new KdTree;
        for (int i = 0; i < cloudf.size(); i++)
            tree->insert(cloudf[i], i);
        DBSCAN ds(MINIMUM_POINTS, EPSILON, cloudf, tree); //Running DBSACN on point cloud 
        ds.run();

        // Iterating clusters and storing according to cluster_id
        for (int i = 1; i < ds.total_clusters; i++)
        {
            vector< PointXYZ <long double> > cur_clus;
            vector< PointXYZ <long double> >::iterator iter;
            for (iter = ds.m_points.begin(); iter != ds.m_points.end(); ++iter)
            {
                if ((*iter).clusterID == i) {
                    cur_clus.push_back(*iter);

                }
            }
            if (cur_clus.size() > MIN_POINTS_IN_CLUSTER && cur_clus.size() < MAX_POINTS_IN_CLUSTER)
            {
                clusters.push_back(cur_clus);
            }

        }

        // Storing the location of clusters and storing PCD format files locally
        for (auto clust : clusters)
        {
            ofstream clusters_a("D:\\plc_try1\\cloud_cluster_" + to_string(c_idx) + ".pcd");
            std::cout << "PointCloud representing the Cluster: " << clust.size() << " data points." << std::endl;

            // storing .pcd files location into a .txt file.
            outfile_1 << "D:\\plc_try1\\cloud_cluster_" << c_idx << ".pcd" << endl;
            //storing the cluster points into .pcd file 
            clusters_a << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n";
            clusters_a << "WIDTH " << clust.size() << endl;
            clusters_a << "HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n";
            clusters_a << "POINTS " << clust.size() << endl;
            clusters_a << "DATA ascii\n";


            for (int i = 0; i < clust.size(); i++)
            {
                clusters_a << clust[i].x << " " << clust[i].y << " " << clust[i].z << " " << 16711680 << endl;
            }
            clusters_a.close();
            c_idx++;
        }
        std::cerr << "Total no. of clusters formed is = " << clusters.size() << std::endl;
        outfile_1.close(); // closing the opened file.

    

    auto cls_e = high_resolution_clock::now();
    auto cls_d = duration_cast<microseconds>(cls_e - cls_s);
    cout << "Time taken by [Clustering] is equal to (in seconds):::::::::::::  " << cls_d.count() / 1e+6 << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by [Full Code] is equal to (in seconds):::::::::::::  " << duration.count() / 1e+6 << endl;

}
