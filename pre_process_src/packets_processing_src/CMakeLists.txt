cmake_minimum_required(VERSION 3.20)
project(ndr_preprocessing)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(PcapPlusPlus REQUIRED)


add_executable(1_filter 1_filter/1_filter.cpp)
add_executable(3_field_extraction 3_field_extraction/3_field_extraction.cpp)
target_link_libraries("1_filter" PUBLIC PcapPlusPlus::Pcap++)
target_link_libraries("3_field_extraction" PUBLIC PcapPlusPlus::Pcap++)
