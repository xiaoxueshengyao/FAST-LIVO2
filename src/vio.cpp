/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::initializeVIO()
{
  // 创建新的子系数地图
  visual_submap = new SubSparseMap;

  // 相机内参
  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();//缩放因子

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  // 图像宽高
  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);
  // imu到相机的 r p 
  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic;
  M3D tmp;
  Jdphi_dR = Rci;
  //  从 imu 坐标系转换到 camera 坐标系下
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp; // 这个作为雅可比后续更新用  p对R的求导

  // 对图像会进行网格化,计算网格行列数
  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  // 网格总数量
  length = grid_n_width * grid_n_height;

  // 使用 投影
  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    // 初始化边界标志位列表
    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    // 相当于是按照grid数量来进行raycast
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        // 每个grid的采样点
        std::vector<V3D> SamplePointsEachGrid;
        // 计算grid的index
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        // 边界
        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        // 计算grid的中心
        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        // 遍历深度,从grid中心一致投影到3.0m的位置,把这些点都放在grid的采样点中
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          // 像素坐标到世界坐标
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        // 添加到所有点的集合中
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  // 启用COLMAP输出
  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  // 初始话grid相关变量
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  // patch大小 默认config是8
  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  // path金字塔层数 config默认是4  64*4
  warp_len = patch_size_total * patch_pyrimid_level;
  // 边界大小
  border = (patch_size_half + 2) * 8;

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

// 重置网格
void VIOManager::resetGrid()
{
  // map相关变量全部重置
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear(); //检索体素
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();  //附加体素
  append_voxel_points.resize(length);

  total_points = 0; //总点数
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

// 计算p到相机图像平面的投影雅可比 p(x,y,z) 对（u，v)的导
void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J)
{
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

/**
 * 从输入图像 img 中提取一个以指定像素坐标 pc 为中心的图像块（patch），并将其存储到 patch_tmp 数组中
 * 通过双线性插值的方法从输入图像中提取以指定像素坐标为中心的图像块，并考虑了图像金字塔层级的影响。
 */
void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

/**
 * 将一个新的 VisualPoint 对象插入到体素地图中
 */
void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  // 从传入的 VisualPoint 对象中提取点在世界坐标系下的三维坐标
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}


/**
 * 该函数的主要目的是根据给定的相机模型、参考像素点、参考三维点、参考法线、当前帧与参考帧之间的变换关系以及参考图像金字塔层级，
 * 计算从参考帧到当前帧的仿射变换矩阵 A_cur_ref。
 * 这个仿射变换矩阵可用于在视觉惯性里程计（VIO）中对图像块进行变形，以实现不同帧之间的特征匹配和跟踪
 */
void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref, const V3D &normal_ref,
                                                  const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

/**
 * 计算从参考帧到当前帧的仿射变换矩阵 A_cur_ref
 */
void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref,
                                        const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                                        Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}


/**
 * 根据给定的仿射变换矩阵，将参考图像中的一个图像块（patch）经过仿射变换后，采样并存储到指定的数组patch中
 * 将参考帧的图像块变换到当前帧的坐标系下，以便进行特征匹配和跟踪
 */
void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

// 根据放射变换矩阵计算最大搜索层级
int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}


/**
 * 计算两个图像块（patch）之间的归一化互相关系数（Normalized Cross - Correlation, NCC）
 * 归一化互相关系数是一种用于衡量两个信号（在这里是图像块）之间相似度的指标
 * 其值范围在 -1 到 1 之间
 * 值越接近 1 表示两个图像块越相似
 * 值越接近 -1 表示两个图像块越不相似
 * 值接近 0 则表示两者之间没有明显的相关性
 */
double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  // 求ref平均
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  // 求cur平局
  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  // 分子和两个分母组成
  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

// 从视觉稀疏地图检索特征
void VIOManager::retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap->reset();// 重置视觉子图

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear(); // 清空子特征地图

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear(); //不用法线的话，清空

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  // 遍历地图点
  for (int i = 0; i < pg.size(); i++)
  {
    // double t0 = omp_get_wtime();

    // 世界系下的点
    V3D pt_w = pg[i].point_w;

    // 计算所在voxel
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    // 在sub_feat_map中找对应的octotree，前面清空了
    auto iter = sub_feat_map.find(position);
    // 没找到直接添加
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; } //如果存在则置0？？这不一样了么

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    // 转到相机系
    V3D pt_c(new_frame_->w2f(pt_w));

    // z坐标为正
    if (pt_c[2] > 0)
    {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      // 相机坐标系转图像坐标系
      px = new_frame_->cam_->world2cam(pt_c);

      // 检查点是否在相机视野内
      if (new_frame_->cam_->isInFrame(px.cast<int>(), border))
      {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0, 255), -1, 8);
        float depth = pt_c[2]; //点的z作为depth
        int col = int(px[0]);
        int row = int(px[1]);
        //构建对应的深度图
        it[width * row + col] = depth; 
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  // 待删除的voxel
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map) //遍历子特征图
  {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    // 查找特征地图对应位置体素
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      // 获取对应体素的所有点
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++) //遍历体素点
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;        // 为空
        if (pt->obs_.size() == 0) continue; // 没观测

        // 这是计算camera frame下的法向量???
        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        // 计算点的方向,也是转到camera坐标系
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        // 如果Z坐标小于0，则继续
        if (dir[2] < 0) continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80 degree 0.08 85 degree

        // 将世界坐标转的点转换为相机坐标
        V2D pc(new_frame_->w2c(pt->pos_));
        // 是否在相机视野内
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
        {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255, 255), -1, 8);
          voxel_in_fov = true;
          // 计算grid索引
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          // 更新网格状态
          grid_num[index] = TYPE_MAP;
          // 观测向量,帧位姿-点位姿
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          // 计算距离
          float cur_dist = obs_vec.norm();
          // 更新一个最小距离和对应点
          if (cur_dist <= map_dist[index])
          {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      // 如果不在视野内,放在待删除的列表里
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  // 投影模块
  if (raycast_en)
  {
    // 遍历素有的grid
    for (int i = 0; i < length; i++)
    {
      // 如果网格已标记为地图或是边界 不操作
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;
      // 遍历每个grid的投影点
      for (const auto &it : rays_with_sample_points[i])
      {
        // 转到世界坐标系
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        // 计算其在world下的voxel位置
        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        // 找到投影点在子地图中的voxel
        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        //找到了则跳出循环, 是说投影线上的点在子图上
        if (corre_sub_feat_map != sub_feat_map.end()) break; 

        // 找在特征地图中的voxel
        auto corre_feat_map = feat_map.find(sample_pos);
        // 找到了
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          // 获取体素点集合
          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          // 遍历体素点
          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            // 点法向量转到frame
            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue; //z要大于0
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17 80 degree 0.08 85 degree

            // world坐标转到camera
            V2D pc(new_frame_->w2c(pt->pos_));

            // 检查是否在视野范围
            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border))
            {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(255, 255, 0), -1, 8); 
              // sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              // 计算所在grid index
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              // 更新每个grid中最小距离和对应索引
              if (cur_dist <= map_dist[index])
              {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          // 如果体素在视野内，则更新子特征地图
          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          // 如果在特征地图中没找到对应的voxel
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          // 查找平面地图中的样本点
          auto iter = plane_map.find(sample_pos);
          // 如果找到
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            // 如果找到,取对应的八叉树
            current_octo = iter->second->find_correspond(sample_point_w);
            // 如果是平面
            if (current_octo->plane_ptr_->is_plane_)
            {
              // 取出平面, 中心, 法向量
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              // 将平面中心点添加到视觉子图中
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  // 子地图删除不再视野范围的点
  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  // 遍历网格
  for (int i = 0; i < length; i++)
  {
    // 对于地图点
    if (grid_num[i] == TYPE_MAP)
    {
      // double t_1 = omp_get_wtime();

      // 拿grid对应的体素点
      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      // 世界系转像素系, 但是貌似计算的矩阵是一样的
      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255), -1, 8); // Green Sparse Align tracked
      // 转相机系
      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      // 遍历patch每个像素
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          // 计算投影点对应深度值
          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          // 特征地图的点 和 patch对应投影点 深度差
          double delta_dist = abs(pt_cam[2] - depth);

          // 认为是深度不连续 
          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      Feature *ref_ftr; //参考特征
      std::vector<float> patch_wrap(warp_len); // 所有pathch大小

      int search_level;
      // 当前参考的仿射矩阵
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        // 光度误差
        float phtometric_errors_min = std::numeric_limits<float>::max();

        // 如果观察数据只有一个
        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_) //如果!=1且没有参考pitch
        {
          // 遍历所有观测
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_; // 获取参考补丁数据
            float phtometric_errors = 0.0;  //初始化
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue; //跳过相同的
              // 拿出缓存的patch
              float *patch_cache = (*itm)->patch_;

              // 计算光度误差, 两两观测
              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            // 均值
            phtometric_errors = phtometric_errors / count;
            // 跟新光度最小值和参考特征
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          // 设置参考patch, obs
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; } //有的话直接用
      }
      else //不用法线
      {
        // 获取最近观测特征
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      if (normal_en)
      {
        // 法向量转camera 并归一化
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
        
        // 参考特征的camera坐标系
        V3D pf(ref_ftr->T_f_w_ * pt->pos_);
        // V3D pf_norm = pf.normalized();
        
        // double cos_theta = norm_vec.dot(pf_norm);
        // if(cos_theta < 0) norm_vec = -norm_vec;
        // if (abs(cos_theta) < 0.08) continue; // 0.5 60 degree 0.34 70 degree 0.17 80 degree 0.08 85 degree

        // 参考特征和当前帧的位姿变换  差
        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        // 计算参考到当前的仿射变换
        getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        // 如果不用法线
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }
      // t_4 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();

      // 遍历patch金字塔层
      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      // 从图像中获取patch
      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++) //计算总误差
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - state->inv_expo_time * patch_buffer[ind]);
      }

      if (ncc_en) // 启用ncc
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      // 误差过大,则继续
      if (error > outlier_threshold * patch_size_total) continue;

      // 体素点加入到视觉子地图
      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error); //添加误差到诗句子图
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);  //添加patch
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_); //逆曝光时间

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  // 视觉子地图总点数
  total_points = visual_submap->voxel_points.size();

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

// 计算雅可比，更新状态
void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;
  
  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    // 如果启用逆组合方法？？
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

/**
 * 根据输入的图像 img 和点云数据 pg，生成视觉地图点并插入到体素地图中
 */
void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  // 过少
  if (pg.size() <= 10) return;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    // 世界坐标
    V3D pt = pg[i].point_w;
    // 像素坐标
    V2D pc(new_frame_->w2c(pt));

    // 检查投影后的像素坐标是否在相机图像帧内，同时考虑边界区域（border）
    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      // 根据像素坐标计算该点所在的网格索引
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      // 如果不是地图点
      if (grid_num[index] != TYPE_MAP)
      {
         // 计算该像素点的 Shi - Tomasi 角点响应值，用于评估该点的特征强度
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
         // 如果当前点的角点响应值大于该网格之前记录的最大响应值
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          // 将当前点存储到该网格对应的追加体素点数组中
          append_voxel_points[index] = pg[i];
          grid_num[index] = TYPE_POINTCLOUD; // 标记类型
        }
      }
    }
  }

  // 遍历视觉子地图中待添加的体素地图点向量
  // 流程和前面一样
  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index])
        {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  // 遍历grid
  for (int i = 0; i < length; i++)
  {
    // 点云类型
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      // 需要添加的点
      pointWithVar pt_var = append_voxel_points[i];
      // world系坐标
      V3D pt = pt_var.point_w;

      // 法向量转到相机系
      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      // 点坐标转到相机系
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      // 计算方向向量和发现两的点成，余弦值
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      // 像素坐标
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      // 从图像中提取该点对应的图像块
      getImagePatch(img, pc, patch, 0);

      // 创建一个新的视觉点对象，传入该点在世界坐标系下的坐标
      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);// 将像素坐标转换为相机坐标系下的归一化方向向量
      // 创建新的特征对象
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      // 将当前帧的 ID 赋值给特征对象的 ID 成员
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      // 将该特征对象添加到视觉点的参考帧列表中
      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      // 根据方向向量与法向量的夹角余弦值，确定视觉点的法向量方向
      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }
      
      // 将当前法向量赋值给视觉点的前一个法向量成员
      pt_new->previous_normal_ = pt_new->normal_;

      // 将该视觉点插入到体素地图中
      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

/**
 * 根据当前帧的图像img更新视觉子地图中的视觉点
 */
void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0) return;  //如果视觉子地图中的总点数为 0

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_; //当前帧位姿
  // 遍历所有视觉点
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;

    // 如果该视觉点已经收敛，删除该视觉点中不被参考的图像块特征
    if (pt->is_converged_)
    { 
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    // 视觉点的三位世界坐标转到像素坐标
    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;
    
    float *patch_temp = new float[patch_size_total];
    // 从当前图像中提取该点对应的图像块
    getImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    // 获取该视觉点最后一次观测的特征
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    // 获取最后一次观测时的相机位姿
    SE3 pose_ref = last_feature->T_f_w_;
    // 计算当前相机位姿与最后一次观测相机位姿之间的相对位姿变换
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    // 计算相对位姿变换的平移部分的范数，即平移距离
    double delta_p = delta_pose.translation().norm();
    // 计算相对位姿变换的旋转部分对应的旋转角度
    double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
    // 如果平移距离大于 0.5 或者旋转角度大于 0.3 弧度，则设置添加标志为 true，表示需要更新该点
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    // 获取最后一次观测时该点在图像中的像素坐标
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    // 维护三维点观测特征的数量，确保不超过 30 个
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      // 找到该视觉点观测特征中得分最小的特征,并在观测中删除
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      update_num += 1;
      update_flag[i] = 1;
      // 相机坐标系的坐标
      Vector3d f = cam->cam2world(pc);
      // 创建一个新的特征对象，包含视觉点、图像块、像素坐标、归一化方向坐标、当前相机位姿和搜索层级信息
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

/**
 * 更新视觉子地图中视觉点的参考图像块
 */
void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    // 视觉地图点筛选
    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    // 计算voxel坐标
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    // 在voxelmap中找到对应voxel
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      // 在八叉树中查找与该点对应的节点
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        // 计算点面距离、绝对值、点到面中心的向量】、投影距离
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

        // 认为在面内
        if (range_dis <= 3 * plane.radius_)
        {
          // 计算法向量相对于点位置和平面法向量的雅可比矩阵
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          // 计算位置不确定性
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;// 累加平面法向量和点协方差的影响

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * plane.normal_);
            // V3D pf(new_frame_->T_f_w_ * pt->pos_);
            // V3D pf_ref(pt->ref_patch->T_f_w_ * pt->pos_);
            // V3D norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);
            
            // 根据点的前一个法向量和平面法向量的点积调整点的法向量方向
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            // 法向量变化量
            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            // 如果法向量更新量小于阈值且观测特征数量大于 10，标记该点为收敛
            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    // 遍历该视觉点的所有观测特征
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      // 初始化归一化互相关（NCC）计算的分子和分母
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      // 世界系转相机系
      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      // 计算点位置向量和法向量的夹角余弦值
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      // 计算当前观测特征图像块的均值
      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      // 遍历该视觉点的所有观测特征，计算 NCC
      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        // 跳过自身
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        // 计算其他观测特征图像块的均值
        float other_mean;
        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        // 计算 NCC 的分子和分母
        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;// 计算得分，得分由 NCC 和夹角余弦值组成

      ref_patch_temp->score_ = score;

      // 更新最大得分和对应patch
      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

/**
 * 预先计算参考图像块的相关雅可比矩阵，并将其存储在 H_sub_inv 矩阵中，
 * 用于描述图像块像素值对相机位姿的偏导数关系。
 */
void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  // H_sub_inv 矩阵的行数，等于总点数乘以图像块的总像素数
  const int H_DIM = total_points * patch_size_total;

  // 存储所有图像块像素对相机位姿（旋转和平移）的雅可比矩阵
  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat; //反对称阵

  // 遍历所有视觉点
  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level); //根据图像金字塔的层级level计算缩放因子

    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_; //获取和patch相关的image

    if (pt == nullptr) continue;

    // 计算当前视觉点相对于参考帧的深度
    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth; //计算当前视觉点在参考帧相机坐标系下的三维坐标
    V2D pc = pt->ref_patch->px_;  //参考patch像素坐标
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix(); // 参考patch的旋转

    // 计算点对投影到相机坐标的雅可比
    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    // 双线性插值权重
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    // 遍历所有patch
    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        // du 和 dv 是u 和 v 方向的梯度
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale); //对梯度值进行缩放，考虑图像金字塔的层级

        //计算图像块像素对相机旋转和平移的雅可比矩阵，链式法则
        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        // 更新在观测雅可比的对应位置
        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}


/**
 * 基于扩展卡尔曼滤波（EKF）算法，利用当前图像 img 和指定的图像金字塔层级 level 来更新视觉惯性里程计（VIO）系统的状态
 * 迭代地计算观测误差和雅可比矩阵，通过最小化误差来优化系统状态，直到满足收敛条件或达到最大迭代次数
 */
void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0) return;
  // 获取旧状态
  StatesGroup old_state = (*state);
  V2D pc; //像素坐标
  MD(1, 2) Jimg; //图像雅可比
  MD(2, 3) Jdpi; //深度雅可比
  MD(1, 3) Jdphi, Jdp, JdR, Jdt; 
  VectorXd z; //观测向量
  MatrixXd H_sub; //观测雅可比
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat; // 坐标反对称阵
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total; //维度，总点数*总patch size

  // 观测数量
  z.resize(H_DIM);
  z.setZero();

  // 观测雅可比维度
  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    // 如果没有参考patch缓存，则计算
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;  //测量数量
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    // 当前帧位姿
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    // 遍历所有视觉点
    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      // 转到像素坐标系
      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale; // 整数部分
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale; //小数
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);  //左上角权重
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);          //右上角权重
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;          //左下
      const float w_ref_br = subpix_u_ref * subpix_v_ref;                  //右下

      // 当前点的patch
      vector<float> P = visual_submap->warp_patch[i];
      
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          // 通过双线性插值计算当前图像块像素值与参考图像块像素值的残差res
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res; //每个patch对应测量向量
          patch_error += res * res; //patch 误差
          // H_sub_inv提取
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0); //对旋转的雅可比
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3); //对平移的雅可比
          // 根据当前状态变换
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          // 更新雅可比
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;// 更新视觉子图中的误差
      error += patch_error; //更新误差和
    }

    error = error / n_meas; //求平均误差

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    // 如果<=上次的误差
    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      // 卡尔曼增益，这套应该是按照fastlio2中改进的公式
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // 误差状态？
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      // 求解
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      // 状态更新
      (*state) += solution;
      // 旋转和位移增量
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      // 增量足够小时认为收敛
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      // 误差比上一次还大，不再更新
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    // 结束迭代
    if (iteration == max_iterations || EKF_end) break; 
  }
}

/**
 * 更新视觉惯性里程计（VIO）系统的状态
 * 结合当前图像 img 和指定的图像金字塔层级 level，迭代直到满足收敛条件或达到最大迭代次数
 */
void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);//取当前状态

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  // 观测维度/数量
  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);//updateStateInverse中时（H_DIM,6）维的
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    // 从当前状态中提取旋转矩阵和位置
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    // 从世界系转到相机系
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    // 点p 对 平移t的求导
    Jdp_dt = Rci * Rwi.transpose();
    
    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);
  
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg; // 图像像素变化相对于图像坐标的偏导数
      MD(2, 3) Jdpi; // 三维点投影到图像平面的偏导数
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;  // 定义旋转、平移等相关的雅可比矩阵

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];  // 获取当前视觉点的搜索层级
      int pyramid_level = level + search_level;            // 计算当前使用的图像金字塔层级
      int scale = (1 << pyramid_level);                    // 缩放
      float inv_scale = 1.0f / scale;

      // 获取当前视觉点的指针
      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      // 将当前视觉点从世界坐标系转换到相机坐标系
      V3D pf = Rcw * pt->pos_ + Pcw;
      // 转到像素坐标
      V2D pc = cam->world2cam(pf);

      // 计算投影雅可比矩阵 Jdpi
      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];// 提取像素坐标的 u 和 v 
      float v_ref = pc[1];
      // 计算像素坐标的整数部分
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      // 计算像素坐标的小数部分
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      // 计算双线性插值的权重
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      // 获取当前视觉点的参考图像块数据
      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i]; // 获取当前视觉点的参考曝光倒数
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      // 遍历当前图像块patch的每一行
      for (int x = 0; x < patch_size; x++)
      {
        // 当前行的启示指针
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          // 当前像素在u 和 v方向的梯度
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          // 对图像的雅可比
          Jimg << du, dv;
          // 考虑曝光时间
          Jimg = Jimg * state->inv_expo_time;
          // 考虑金字塔层级的缩放
          Jimg = Jimg * inv_scale;
          // 计算旋转雅可比
          Jdphi = Jimg * Jdpi * p_hat;
          // 平移
          Jdp = -Jimg * Jdpi;
          // 旋转更新的雅可比矩阵 JdR
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          // 平移更新的雅可比矩阵 Jdt
          Jdt = Jdp * Jdp_dt;

          // 计算当前像素的插值值
          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          // 计算当前像素的残差
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];
          
          // 将残差存储到观测向量 z 中
          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res; //累加到patch的误差中
          n_meas += 1;
          
          // 开启曝光估计的话，将旋转、平移和当前像素值存储到观测雅可比矩阵 H_sub 中
          if (exposure_estimate_en) 
          { 
            H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; 
          } else { 
            // 否则只存储旋转和平移的雅可比矩阵
            H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; 3
          } 
        }
      }
      visual_submap->errors[i] = patch_error;// 保存当前视觉点的图像块误差
      error += patch_error; //累加
    }

    error = error / n_meas;
    
    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

     // 如果当前误差小于等于上一次的误差
    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      // 计算 H_sub_T * H_sub 并存储到 H_T_H 中
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      // 卡尔曼增益
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      // Hz
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      // 计算状态预测值与当前状态的差值
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
       // 计算状态更新量
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      // 更新状态
      (*state) += solution;
      // 获取增量
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      // 判断收敛
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

// 更新当前状态
void VIOManager::updateFrameState(StatesGroup state)
{
  // 获取当前状态
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  // 当前帧的旋转 world系下
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3(Rcw, Pcw); //当前帧坐标
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i), cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size * i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " + std::to_string(total_points);
  // cv::Point2f origin;
  // origin.x = img_cp.cols - 110;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap()
{
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";
  
  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);
  
  Eigen::Quaterniond q(new_frame_->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

//更新视觉地图点 视觉地图点的处理和更新是论文重点
/**
 * @brief: 视觉帧处理流程
 * 
 * @img: 当前视觉帧
 * @pg: 激光地图点+cov
 * @feat_map: voxel map
 * @img_time: lio更新时间到第一帧的时间
 */
void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  // 如果尺寸不匹配
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
  }
  // 复制
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  // 转灰度图
  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img)); //创建新的帧
  // 更新帧的状态
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();

  // 从视觉稀疏地图中检索特征
  retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  // 计算雅可比,做ekf更新
  computeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  // 生成视觉地图点
  generateVisualMapPoints(img, pg);

  double t4 = omp_get_wtime();
  
  // 画被跟踪的点
  plotTrackedPoints();

  // 如果启用绘图标志，则将参考补丁投影到当前帧
  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  // 更新视觉地图点
  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();
  
  if(colmap_output_en)  dumpDataForColmap();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
  // cv::imwrite("/home/chunran/Desktop/raycasting/" + std::to_string(new_frame_->id_) + ".png", img_cp);
}
