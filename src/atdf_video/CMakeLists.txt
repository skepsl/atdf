cmake_minimum_required(VERSION 3.0.2)
project(atdf_video)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  pcl_ros
  pcl_conversions
  laser_geometry
  geometry_msgs
  nav_msgs
  visualization_msgs
  actionlib
  actionlib_msgs
  move_base_msgs
  tf
  tf2_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS src/atdf_video.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# Standalone executable helpers, private to this ROS package. No Python
# package is declared: there is no src/atdf_video/__init__.py to install.
configure_file(src/atdf_rviz.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/atdf_rviz.py COPYONLY)
configure_file(src/atdf_sionna.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/atdf_sionna.py COPYONLY)
configure_file(src/atdf_timing.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/atdf_timing.py COPYONLY)
configure_file(src/atdf_artifacts.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/atdf_artifacts.py COPYONLY)
configure_file(src/atdf_plot.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/atdf_plot.py COPYONLY)
install(FILES src/atdf_rviz.py src/atdf_sionna.py src/atdf_timing.py src/atdf_artifacts.py src/atdf_plot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config maps rviz checkpoints
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
