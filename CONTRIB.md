# SSRLCV Contributer Guide

# Doxygen Documentation

# Style Guide


# Adding Feature Descriptors

1. Add Descriptor to Feature.cuh & Feature.cu
2. Make sure to include distProtocol(D otherDescriptor) as a member function for MatchFactory compatibility
3. Avoid using pointers within Descriptor to avoid cuda memory issues and compatibility with FeatureFactory and MatchFactory