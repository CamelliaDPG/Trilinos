// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
//  Intrepid2_VectorData.cpp
//
//  Created by Roberts, Nathan V on 5/13/25.
//

#include "Intrepid2_VectorData.hpp"

using DefaultDeviceType = Kokkos::DefaultExecutionSpace::device_type;
template class Intrepid2::VectorData<double,DefaultDeviceType>;
