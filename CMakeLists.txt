
add_executable(detect_bush src/detect_bush.cpp)
target_link_libraries(detect_bush ${catkin_LIBRARIES}${PCL_LIBRARIES})





install(
TARGETS
   detect_bush
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


