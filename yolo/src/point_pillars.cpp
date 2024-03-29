#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
// #include <sklearn/base.h>
// #include <sklearn/neighbors/kneighbors_classifier.h>
namespace py = pybind11;

struct IntPairHash {
  std::size_t operator()(const std::pair<uint32_t, uint32_t> &p) const {
    assert(sizeof(std::size_t)>=8);
    //Shift first integer over to make room for the second integer. The two are
    //then packed side by side.
    return (((uint64_t)p.first)<<32) | ((uint64_t)p.second);
  }
};

struct PillarPoint {
    float x;
    float y;
    float z;
    float intensity;
    float xc;
    float yc;
    float zc;
};

pybind11::tuple createPillars(pybind11::array_t<float> points,
                              int maxPointsPerPillar,
                              int maxPillars,
                              float xStep,
                              float yStep,
                              float xMin,
                              float xMax,
                              float yMin,
                              float yMax,
                              float zMin,
                              float zMax,
                              bool printTime = true)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    


    if (points.ndim() != 2 || points.shape()[1] != 4)
    {
        throw std::runtime_error("numpy array with shape (n, 4) expected (n being the number of points)");
    }

    // Hasmap for holding indices
    std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<PillarPoint>, IntPairHash> map;

    for (int i = 0; i < points.shape()[0]; ++i)
    {
        if ((points.at(i, 0) < xMin) || (points.at(i, 0) >= xMax) || \
            (points.at(i, 1) < yMin) || (points.at(i, 1) >= yMax) || \
            (points.at(i, 2) < zMin) || (points.at(i, 2) >= zMax))
        {
            continue;
        }

        // Point-cloud discretization on the x-y plane
        auto xIndex = static_cast<uint32_t>(std::floor((points.at(i, 0) - xMin) / xStep));
        auto yIndex = static_cast<uint32_t>(std::floor((points.at(i, 1) - yMin) / yStep));

        PillarPoint p = {
            points.at(i, 0),
            points.at(i, 1),
            points.at(i, 2),
            points.at(i, 3),
            0,
            0,
            0,
        };

        // Pushing the PointPillar object into the hasmap, hash will be generated on the basis of discritized x, y coordinate
        map[{xIndex, yIndex}].emplace_back(p);
    }

    pybind11::array_t<float> tensor;
    pybind11::array_t<int> indices;

    tensor.resize({1, maxPillars, maxPointsPerPillar, 7});
    indices.resize({1, maxPillars, 3}); // Will hold the discretized indices of the pillar

    int pillarId = 0;
    for (auto& pair: map) // Iterating through the hash-map
    {
        if (pillarId >= maxPillars)
        {
            // Number of populated pillars exceeds the maximum allowed number of pillars
            break;
        }

        // if (pair.second.size()<3){continue;}

        float xMean = 0;
        float yMean = 0;
        float zMean = 0;
        // Iterating through all the points for current hash (or pillar) for mean coordinate calculation
        for (const auto& p: pair.second) // pair.first -> hash, pair.second -> value
        {
            xMean += p.x;
            yMean += p.y;
            zMean += p.z;
        }
        xMean /= pair.second.size();
        yMean /= pair.second.size();
        zMean /= pair.second.size();

        // Updating distance from calculated mean for each point of current hash (pillar)
        for (auto& p: pair.second)
        {
            p.xc = p.x - xMean;
            p.yc = p.y - yMean;
            p.zc = p.z - zMean;
        }

        // Discretizing the mean coordinates of the current hash (pillar)
        auto xIndex = static_cast<int>(std::floor((xMean - xMin) / xStep));
        auto yIndex = static_cast<int>(std::floor((yMean - yMin) / yStep));
        auto zMid   = (zMax - zMin) * 0.5f;

        // Updating the indices of the pillar for current pillar id
        indices.mutable_at(0, pillarId, 1) = xIndex;
        indices.mutable_at(0, pillarId, 2) = yIndex;

        int pointId = 0;
        for (const auto& p: pair.second) // Iterating through all the points of current hash (pillar)
        {
            // Point data population inside the pillar
            if (pointId >= maxPointsPerPillar)
            {
                // Number of populated points exceeds the maximum allowed number of points per pillar
                break;
            }

            // Raw LiDAR point data
            // tensor.mutable_at(0, pillarId, pointId, 0) = p.x;
            // tensor.mutable_at(0, pillarId, pointId, 1) = p.y;
            tensor.mutable_at(0, pillarId, pointId, 0) = p.z;
            tensor.mutable_at(0, pillarId, pointId, 1) = p.intensity;
            // Distance from the aithrmetic mean
            tensor.mutable_at(0, pillarId, pointId, 2) = p.xc;
            tensor.mutable_at(0, pillarId, pointId, 3) = p.yc;
            tensor.mutable_at(0, pillarId, pointId, 4) = p.zc;
            // Offset from the pillar center
            tensor.mutable_at(0, pillarId, pointId, 5) = ((p.x - xMin) / xStep) - xIndex;
            tensor.mutable_at(0, pillarId, pointId, 6) = ((p.y - yMin) / yStep) - yIndex;

            pointId++;
        }

        // point size? point correlation 

        pillarId++;
    }

    pybind11::tuple result = pybind11::make_tuple(tensor, indices);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    // if (printTime)
    // std::cout << "createPillars took: " << static_cast<float>(duration) / 1e6 << " seconds" << std::endl;

    return result;
}

struct BoundingBox3D
{
    float x;
    float y;
    float z;
    float length;
    float width;
    float height;
    float yaw;
    float classId;
};

struct Point2D {
    float x;
    float y;
};

typedef std::vector<Point2D> Polyline2D;

// Returns x-value of point of intersection of two lines
float xIntersect(float x1, float y1, float x2, float y2,
                     float x3, float y3, float x4, float y4)
{
    float num = (x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4);
    float den = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4);
    return num/den;
}

// Returns y-value of point of intersection of two lines
float yIntersect(float x1, float y1, float x2, float y2,
                     float x3, float y3, float x4, float y4)
{
    float num = (x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4);
    float den = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4);
    return num/den;
}

// Returns area of polygon using the shoelace method
float polygonArea(const Polyline2D &polygon)
{
    float area = 0.0;

    size_t j = polygon.size()-1;
    for (size_t i = 0; i < polygon.size(); i++)
    {
        area += (polygon[j].x + polygon[i].x) * (polygon[j].y - polygon[i].y);
        j = i;  // j is previous vertex to i
    }

    return std::abs(area / 2.0); // Return absolute value
}

float rotatedX(float x, float y, float angle)
{
    return x * std::cos(angle) - y * std::sin(angle);
}

float rotatedY(float x, float y, float angle)
{
    return x * std::sin(angle) + y * std::cos(angle);
}

// Construct bounding box in 2D, coordinates are returned in clockwise order
Polyline2D boundingBox3DToTopDown(const BoundingBox3D &box1)
{
    Polyline2D box;
    box.push_back({rotatedX(-0.5 * box1.width, 0.5 * box1.length,
                            box1.yaw) + box1.x,
                   rotatedY(-0.5 * box1.width, 0.5 * box1.length,
                            box1.yaw) + box1.y});

    box.push_back({rotatedX(0.5 * box1.width, 0.5 * box1.length,
                            box1.yaw) + box1.x,
                   rotatedY(0.5 * box1.width, 0.5 * box1.length,
                            box1.yaw) + box1.y});

    box.push_back({rotatedX(0.5 * box1.width, -0.5 * box1.length,
                            box1.yaw) + box1.x,
                   rotatedY(0.5 * box1.width, -0.5 * box1.length,
                            box1.yaw) + box1.y});

    box.push_back({rotatedX(-0.5 * box1.width, -0.5 * box1.length,
                            box1.yaw) + box1.x,
                   rotatedY(-0.5 * box1.width, -0.5 * box1.length,
                            box1.yaw) + box1.y});

    return box;
}

// This functions clips all the edges w.r.t one Clip edge of clipping area
// Returns a clipped polygon...
Polyline2D clip(const Polyline2D &poly_points,
                float x1,
                float y1,
                float x2,
                float y2)
{
    Polyline2D new_points;

    for (size_t i = 0; i < poly_points.size(); i++)
    {
        // (ix,iy),(kx,ky) are the co-ordinate values of the points
        // i and k form a line in polygon
        size_t k = (i+1) % poly_points.size();
        float ix = poly_points[i].x, iy = poly_points[i].y;
        float kx = poly_points[k].x, ky = poly_points[k].y;

        // Calculating position of first point w.r.t. clipper line
        float i_pos = (x2-x1) * (iy-y1) - (y2-y1) * (ix-x1);

        // Calculating position of second point w.r.t. clipper line
        float k_pos = (x2-x1) * (ky-y1) - (y2-y1) * (kx-x1);

        // Case 1 : When both points are inside
        if (i_pos < 0  && k_pos < 0)
        {
            //Only second point is added
            new_points.push_back({kx,ky});
        }

            // Case 2: When only first point is outside
        else if (i_pos >= 0  && k_pos < 0)
        {
            // Point of intersection with edge
            // and the second point is added
            new_points.push_back({xIntersect(x1, y1, x2, y2, ix, iy, kx, ky),
                                  yIntersect(x1, y1, x2, y2, ix, iy, kx, ky)});
            new_points.push_back({kx,ky});

        }

            // Case 3: When only second point is outside
        else if (i_pos < 0  && k_pos >= 0)
        {
            //Only point of intersection with edge is added
            new_points.push_back({xIntersect(x1, y1, x2, y2, ix, iy, kx, ky),
                                  yIntersect(x1, y1, x2, y2, ix, iy, kx, ky)});

        }
            // Case 4: When both points are outside
        else
        {
            //No points are added
        }
    }

    return new_points;
}

// Implements Sutherland–Hodgman algorithm
// Returns a polygon with the intersection between two polygons.
Polyline2D sutherlandHodgmanClip(const Polyline2D &poly_points_vector,
                                 const Polyline2D &clipper_points)
{
    Polyline2D clipped_poly_points_vector = poly_points_vector;
    for (size_t i=0; i<clipper_points.size(); i++)
    {
        size_t k = (i+1) % clipper_points.size(); //i and k are two consecutive indexes

        // We pass the current array of vertices, and the end points of the selected clipper line
        clipped_poly_points_vector = clip(clipped_poly_points_vector, clipper_points[i].x, clipper_points[i].y,
                                          clipper_points[k].x, clipper_points[k].y);
    }
    return clipped_poly_points_vector;
}

float center_dist(const BoundingBox3D& box1,
          const BoundingBox3D& box2)
{
    float dist=std::pow(box2.x - box1.x, 2) + std::pow(box1.y - box2.y, 2);
    return dist;
}

float c_diag(const  Polyline2D & box1,
          const  Polyline2D & box2)
{
    std::vector<float> xmin_values = { box1[0].x,box1[3].x,box2[0].x,box2[3].x};
    std::vector<float> xmax_values = { box1[1].x,box1[1].x,box2[1].x,box2[2].x};
    std::vector<float> ymin_values = { box1[2].y,box1[3].y,box2[2].y,box2[3].y};
    std::vector<float> ymax_values = { box1[1].y,box1[0].y,box2[1].y,box2[0].y};

    auto min_x_ele = std::min_element(xmin_values.begin(), xmin_values.end());
    int min_x = *min_x_ele;
    auto min_y_ele = std::min_element(ymin_values.begin(), ymin_values.end());
    int min_y = *min_y_ele;
    auto max_x_ele = std::max_element(xmax_values.begin(), xmax_values.end());
    int max_x = *max_x_ele;
    auto max_y_ele = std::max_element(ymax_values.begin(), ymax_values.end());
    int max_y = *max_y_ele;

    float dist=std::pow(max_x- min_x, 2) + std::pow(max_y - min_y, 2);
    return dist;
}

// Calculates the IOU between two bounding boxes.
float iou(const BoundingBox3D& box1,
          const BoundingBox3D& box2)
{
    const auto& box_as_vector = boundingBox3DToTopDown(box1);
    const auto& box_as_vector_2 = boundingBox3DToTopDown(box2);
    const auto& clipped_vector = sutherlandHodgmanClip(box_as_vector, box_as_vector_2);

    float area_poly1 = polygonArea(box_as_vector);
    float area_poly2 = polygonArea(box_as_vector_2);
    float area_overlap = polygonArea(clipped_vector);

    return area_overlap / (area_poly1 + area_poly2 - area_overlap);
}

float ciou(const BoundingBox3D& box1,
          const BoundingBox3D& box2)
{
    const auto& box_as_vector = boundingBox3DToTopDown(box1);
    const auto& box_as_vector_2 = boundingBox3DToTopDown(box2);
    const auto& clipped_vector = sutherlandHodgmanClip(box_as_vector, box_as_vector_2);
    const auto& dist=center_dist(box1,box2);
    const auto& c=c_diag(box_as_vector,box_as_vector_2);

    float area_poly1 = polygonArea(box_as_vector);
    float area_poly2 = polygonArea(box_as_vector_2);
    float area_overlap = polygonArea(clipped_vector);

    return (area_overlap / (area_poly1 + area_poly2 - area_overlap))-dist/c;
}

int clip(int n, int lower, int upper) {
  return std::max(lower, std::min(n, upper));
}

float align_iou(const BoundingBox3D& box1,
          const BoundingBox3D& box2)
{
    // Compute the coordinates of the four corners of each rectangle
    double x1_min = box1.x- box1.length/2, x1_max = box1.x+ box1.length/2;
    double y1_min = box1.y- box1.width/2, y1_max = box1.y+ box1.width/2;
    double x2_min = box2.x- box2.length/2, x2_max = box2.x+ box2.length/2;
    double y2_min = box2.y- box2.width/2, y2_max = box2.y+ box2.width/2;

    // Compute the overlap area between the two rectangles
    double x_overlap = std::max(0.0, std::min(x1_max, x2_max) - std::max(x1_min, x2_min));
    double y_overlap = std::max(0.0, std::min(y1_max, y2_max) - std::max(y1_min, y2_min));
    double overlap_area = x_overlap * y_overlap;

    // Compute the union area of the two rectangles
    double area1 = box1.length * box1.width;
    double area2 = box2.length * box2.width;
    double union_area = area1 + area2 - overlap_area;

    // Compute the IoU
    float iou = overlap_area / union_area;

    // Return the IoU value
    return iou;
}

float iouraw(
    const pybind11::array_t<float>& labelBoxList,
    const pybind11::array_t<float>& anchorBoxList)
{
    BoundingBox3D labelBox = {};
    labelBox.x = labelBoxList.at(0);
    labelBox.y = labelBoxList.at(1);
    labelBox.z = labelBoxList.at(2);
    labelBox.length = labelBoxList.at(3);
    labelBox.width = labelBoxList.at(4);
    labelBox.height = labelBoxList.at(5);
    labelBox.yaw = labelBoxList.at(6);

    BoundingBox3D anchorBox = {};
    anchorBox.x = anchorBoxList.at(0);
    anchorBox.y = anchorBoxList.at(1);
    anchorBox.z = anchorBoxList.at(2);
    anchorBox.length = anchorBoxList.at(3);
    anchorBox.width = anchorBoxList.at(4);
    anchorBox.height = anchorBoxList.at(5);
    anchorBox.yaw = anchorBoxList.at(6);

    return iou(labelBox,anchorBox);

}

float ciouraw(
    const pybind11::array_t<float>& labelBoxList,
    const pybind11::array_t<float>& anchorBoxList)
{
    BoundingBox3D labelBox = {};
    labelBox.x = labelBoxList.at(0);
    labelBox.y = labelBoxList.at(1);
    labelBox.z = labelBoxList.at(2);
    labelBox.length = labelBoxList.at(3);
    labelBox.width = labelBoxList.at(4);
    labelBox.height = labelBoxList.at(5);
    labelBox.yaw = labelBoxList.at(6);

    BoundingBox3D anchorBox = {};
    anchorBox.x = anchorBoxList.at(0);
    anchorBox.y = anchorBoxList.at(1);
    anchorBox.z = anchorBoxList.at(2);
    anchorBox.length = anchorBoxList.at(3);
    anchorBox.width = anchorBoxList.at(4);
    anchorBox.height = anchorBoxList.at(5);
    anchorBox.yaw = anchorBoxList.at(6);

    if (labelBox.length==0) return 0;
    return ciou(labelBox,anchorBox);

}
std::tuple<pybind11::array_t<float>, int, int> createPillarsTarget(const pybind11::array_t<float>& objectPositions,
                                             const pybind11::array_t<float>& objectDimensions,
                                             const pybind11::array_t<float>& objectYaws,
                                             const pybind11::array_t<int>& objectClassIds,
                                             const pybind11::array_t<float>& anchorDimensions,
                                             const pybind11::array_t<float>& anchorZHeights,
                                             const pybind11::array_t<float>& anchorYaws,
                                             float positiveThreshold,
                                             float negativeThreshold,
                                             int nbClasses,
                                             int downscalingFactor,
                                             float xStep,
                                             float yStep,
                                             float xMin,
                                             float xMax,
                                             float yMin,
                                             float yMax,
                                             float zMin,
                                             float zMax,
                                             bool printTime = true)
{
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // getting downsampled grid size
    const auto xSize = static_cast<int>(std::floor((xMax - xMin) / (xStep * downscalingFactor)));
//     py::print("xSize", xSize);
    const auto ySize = static_cast<int>(std::floor((yMax - yMin) / (yStep * downscalingFactor)));
//     py::print("ySize", ySize);

    // sk::KNeighborsClassifier clf;
    // clf.load("knn_model.joblib");


    const int nbAnchors = anchorDimensions.shape()[0]; //4 Number of anchors
//     py::print("nbAnchors", nbAnchors);
//     Anchor length

    if (nbAnchors <= 0)
    {
        throw std::runtime_error("Anchor length is zero");
    }

    const int nbObjects = objectDimensions.shape()[0]; //6 Number of labels inside a label.txt file
//     BB dimensions from the label file
    if (nbObjects <= 0)
    {
        throw std::runtime_error("Object length is zero");
    }
//     py::print("nbObjects", nbObjects);

    // parse numpy arrays
//     Preparing the anchor bounding box
    std::vector<BoundingBox3D> anchorBoxes = {};
    std::vector<float> anchorDiagonals;
    for (int i = 0; i < nbAnchors; ++i)
    {
        BoundingBox3D anchorBox = {};
        anchorBox.x = 0;
        anchorBox.y = 0;
        anchorBox.length = anchorDimensions.at(i, 0);
        anchorBox.width = anchorDimensions.at(i, 1);
        anchorBox.height = anchorDimensions.at(i, 2);
        anchorBox.z = anchorZHeights.at(i);
        anchorBox.yaw = anchorYaws.at(i);
        anchorBoxes.emplace_back(anchorBox); // Appends a new anchorBox to the AnchorBoxes container
        // Note that anchor box doesn't have a classId as of now.
        // Length of (width,length) axis diagonal.
        anchorDiagonals.emplace_back(std::sqrt(std::pow(anchorBox.width, 2) + std::pow(anchorBox.length, 2)));
    }

//     Preparing the label bounding box
    std::vector<BoundingBox3D> labelBoxes = {};
    for (int i = 0; i < nbObjects; ++i)
    {
        float x = objectPositions.at(i, 0);
        float y = objectPositions.at(i, 1);
        if (x < xMin | x > xMax | y < yMin | y > yMax)
        {
            continue;
        }
        BoundingBox3D labelBox = {};
        labelBox.x = x;
        labelBox.y = y;
        labelBox.z = objectPositions.at(i, 2);
        labelBox.length = objectDimensions.at(i, 0);
        labelBox.width = objectDimensions.at(i, 1);
        labelBox.height = objectDimensions.at(i, 2);
        labelBox.yaw = objectYaws.at(i);
        labelBox.classId = objectClassIds.at(i);
        labelBoxes.emplace_back(labelBox);
    }

    pybind11::array_t<float> tensor;
    tensor.resize({nbObjects, xSize, ySize, nbAnchors, 10}); //Tensor of size (6,252,252,4,10) for first file
    
    // getting tensor information as defined in Python buffer protocol specification 
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
    pybind11::buffer_info tensor_buffer = tensor.request();
    float *ptr1 = (float *) tensor_buffer.ptr;
    // Zero filling the tensor. Every element is presently zero
    for (size_t idx = 0; idx < nbObjects * xSize * ySize * nbAnchors * 10; idx++)
    {
        ptr1[idx] = 0;
    }

    int posCnt = 0;
    int negCnt = 0;
    int occu=0;
    int objectCount = 0;
    if (printTime)
    {
//         std::cout << "Received " << labelBoxes.size() << " objects" << std::endl;
//         py::print("Received "+str(labelBoxes.size())+" objects");
    }
    for (const auto& labelBox: labelBoxes) //For every label box which is a 3d bounding box
    {
        // zone-in on potential spatial area of interest
        // Length of (width,length) axis diagonal.
        float objectDiameter = std::sqrt(std::pow(labelBox.width, 2) + std::pow(labelBox.length, 2))/2;
        // Offset = Number of grid boxes that can fit on the object diameter
        const auto x_offset = static_cast<int>(std::ceil(objectDiameter / (xStep * downscalingFactor)));
        const auto y_offset = static_cast<int>(std::ceil(objectDiameter / (yStep * downscalingFactor)));
        // Xc = Number of grid boxes that can fit between Xmin (Ymin) and Label's x (y) coordinate
        const auto xC = static_cast<int>(std::floor((labelBox.x - xMin) / (xStep * downscalingFactor)));
        const auto yC = static_cast<int>(std::floor((labelBox.y - yMin) / (yStep * downscalingFactor)));
        // X(Y)Start = Start from Xc (Yc) - Number of boxes in object's diameter.
        // For example the object is located at 5 unites and is 2 unites long. Then X(Y)start will begin
        // the search from 3
        const auto xStart = clip(xC - x_offset, 0, xSize);
        const auto yStart = clip(yC - y_offset, 0, ySize);
        // Similarly end the search at 8 units. Because the object cannot extend beyond that.
        const auto xEnd = clip(xC + x_offset, 0, xSize);
        const auto yEnd = clip(yC + y_offset, 0, ySize);

        float maxIou = 0;
        BoundingBox3D bestAnchor = {};
        int bestAnchorId = 0;
        int bestAnchor_xId = 0;
        int bestAnchor_yId = 0;
        for (int xId = xStart; xId < xEnd; xId++) // Iterate through every box within search diameter
            // In our example case, from 3 till 8
        {
            // Getting the real world x coordinate
            const float x = xId * xStep * downscalingFactor + xMin;
            for (int yId = yStart; yId < yEnd; yId++) // Iterate through every box within search diamter in y axis
            {
                // Get the real world y coordinates
                const float y = yId * yStep * downscalingFactor + yMin;
                int anchorCount = 0;
                for (auto& anchorBox: anchorBoxes) // For every anchor box (4 in our case)
                    // Note that we are checking every anchor box for every label in the file
                {
                    anchorBox.x = x; // Assign the real world x and y coordinate to the anchor box
                    anchorBox.y = y; // Note that anchor boxes originally didn't have Xs and Ys.
                    // anchorBox.yaw=clf.predict([x,y])
                    // This is because we need to check them along the X-Y grid.
                    // However, they did have a z value attached to them. 

                    // const float iouOverlap = iou(anchorBox, labelBox); // Get IOU between two 3D boxes.
                    const float iouOverlap = iou(anchorBox, labelBox); // Get IOU between two 3D boxes.

                    if (maxIou < iouOverlap)
                    {
                        maxIou = iouOverlap;
                        bestAnchor = anchorBox;
                        bestAnchorId = anchorCount;
                        bestAnchor_xId = xId;
                        bestAnchor_yId = yId;
                    }

                    if (iouOverlap > positiveThreshold) // Accept the Anchor. Add the anchor details to the tensor.
                    {
                        // Tensor at CurrentObject Id, xth grid cell, yth grid cell, currentAnchor, 0
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = 1; 

                        auto diag = anchorDiagonals[anchorCount];
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 1) = (labelBox.x - anchorBox.x) / diag; // delta x,y,z
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 2) = (labelBox.y - anchorBox.y) / diag;
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 3) = (labelBox.z - anchorBox.z) / anchorBox.height;

                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 4) = std::log(labelBox.length / anchorBox.length); // delta l,w,h
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 5) = std::log(labelBox.width / anchorBox.width);
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 6) = std::log(labelBox.height / anchorBox.height);

                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 7) = (labelBox.yaw - anchorBox.yaw); //delta yaw
                        if (((-0.5 * M_PI) < labelBox.yaw) && (labelBox.yaw <= (0.5 * M_PI)))
                        {
                            tensor.mutable_at(objectCount, xId, yId, anchorCount, 8) = 1; 
                        }
                        else
                        {
                            tensor.mutable_at(objectCount, xId, yId, anchorCount, 8) = 0;
                        }

                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 9) = labelBox.classId;
                        occu++;

                    }
                    else if (iouOverlap < negativeThreshold)
                    {
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = 0;
                    }
                    else
                    {
                        tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = -1;
                    }

                    anchorCount++;
                }
            }
        }

        if (maxIou < positiveThreshold) // Comparing maxIOU for that object obtained after checking with every anchor box
            // If none of the anchors passed the threshold, then we place the best anchor details for that object. 
        {
            negCnt++;
            if (printTime)
            {
//                 std::cout << "\nThere was no sufficiently overlapping anchor anywhere for object " << objectCount << std::endl;
//                 py::print("There was no sufficiently overlapping anchor anywhere for object " +str(objectCount));
//                 std::cout << "Best IOU was " << maxIou << ". Adding the best location regardless of threshold." << std::endl;
//                 py::print("Best IOU was "+str(maxIou)+" Adding the best location regardless of threshold");
            }

            // const auto xId = static_cast<int>(std::floor((labelBox.x - xMin) / (xStep * downscalingFactor)));
            // const auto yId = static_cast<int>(std::floor((labelBox.y - yMin) / (yStep * downscalingFactor)));

            const auto xId = bestAnchor_xId;
            const auto yId = bestAnchor_yId;
            const float diag = std::sqrt(std::pow(bestAnchor.width, 2) + std::pow(bestAnchor.length, 2));

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 0) = 1;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 1) = (labelBox.x - bestAnchor.x) / diag;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 2) = (labelBox.y - bestAnchor.y) / diag;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 3) = (labelBox.z - bestAnchor.z) / bestAnchor.height;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 4) = std::log(labelBox.length / bestAnchor.length);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 5) = std::log(labelBox.width / bestAnchor.width);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 6) = std::log(labelBox.height / bestAnchor.height);

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 7) = (labelBox.yaw - bestAnchor.yaw);
            if (((-0.5 * M_PI) < labelBox.yaw) && (labelBox.yaw <= (0.5 * M_PI)))
            {
                tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 8) = 1;
            }
            else
            {
                tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 8) = 0;
            }
//             Class id is the classification label (0,1,2,3)
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 9) = labelBox.classId;
            occu++;
        }
        else
        {
            posCnt++;
            if (printTime)
            {
            std::cout << "\nAt least 1 anchor was positively matched for object " << objectCount << std::endl;
            std::cout << "Best IOU was " << maxIou << "." << std::endl;
            }
        }

        objectCount++;
        

    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    // if (printTime)
    // std::cout << "createPillarsTarget took: " << static_cast<float>(duration) / 1e6 << " seconds" << std::endl;
    // std::cout << "Number of posCnt " << occu << "." << std::endl;
    return std::make_tuple(tensor, posCnt, negCnt);
}


PYBIND11_MODULE(point_pillars, m)
{
    m.def("createPillars", &createPillars, "Runs function to create point pillars input tensors");
    m.def("createPillarsTarget", &createPillarsTarget, "Runs function to create point pillars output ground truth");
    m.def("iouraw", &iouraw, "Runs function to calculate IOU");
    m.def("ciouraw", &ciouraw, "Runs function to calculate CIOU");


}