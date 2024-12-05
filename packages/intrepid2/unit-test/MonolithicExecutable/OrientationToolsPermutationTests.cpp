// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   OrientationToolsPermutationTests.cpp
    \brief  Tests to verify various permutation-handling methods of OrientationTools.
    \author Created by N.V. Roberts.
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_CellTopology.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_HierarchicalBasisFamily.hpp"
#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_SerendipityBasisFamily.hpp"

#include "Kokkos_Core.hpp"

namespace
{
  using CellTopoPtr = Teuchos::RCP<::Intrepid2::CellTopology>;

  using namespace Intrepid2;
  
  using DeviceType = DefaultTestDeviceType;
  using ExecutionSpace = typename DeviceType::execution_space;
  using Scalar = double;

  ordinal_type ortMax(CellTopoPtr cellTopo)
  {
    cellTopo->
    auto cellTopologyKey = cellTopo.getBaseKey();
    switch (cellTopologyKey) {
      case shards::Node::key:             return 0;
      case shards::Line<2>::key:          return 1;
      case shards::Triangle<3>::key:      return 6;
      case shards::Quadrilateral<4>::key: return 8;
      default: {
        INTREPID2_TEST_FOR_EXCEPTION( true, std::invalid_argument,
                                     ">>> ERROR (ortMax()): unsupported cell topology.");
      }
    }
    return -1;
  }

  void testOrientationsArePermutations(const Intrepid2::CellTopology &cellTopo, BasisPtr<DeviceType,Scalar,Scalar> basis,
                                     Teuchos::FancyOStream &out, bool &success)
  {
    auto maxOrt = ortMax(cellTopo.getSubcell(2,0));
  }
  
  TEUCHOS_UNIT_TEST( OrientationTools, OrientationsArePermutations_Tet )
  {
    shards::CellTopology shardsTopo = shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() );
    Intrepid2::CellTopology cellTopo(shardsTopo, 0);
    
    int edgeDim = 1;
    int faceDim = 2;
    
    cellTopo.getSubcell(<#int scdim#>, <#int scord#>)
  }
} // namespace
