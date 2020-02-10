# SSRLCV Contributor Guide

# Style Guide
## Make sure to add Doxygen comments!
* add todos in doxygen so that issues can be made periodically
## Keep within Factory organization
* FeatureFactory
* MatchFactory
* PointCloudFactory
* MeshFactory
## Header and Code organization
* C++ & C declarations and implementations above the same for CUDA

# Adding Feature Descriptors
1. Add Descriptor to Feature.cuh & Feature.cu
2. Make sure to include distProtocol(D otherDescriptor) as a member function for MatchFactory compatibility
3. Avoid using pointers within Descriptor to avoid cuda memory issues and compatibility with FeatureFactory and MatchFactory