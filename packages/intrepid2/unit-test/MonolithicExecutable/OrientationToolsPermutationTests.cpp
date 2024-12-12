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

#include "Intrepid2_TestUtils.hpp"

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
    auto cellTopologyKey = cellTopo->getKey().first; // key is (shardsKey, numTensorExtrusions)
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
    // 1. Check that the identity orientation (0's for every edge and face) is a permutation
    
    // 2. Iterate through each possible face/edge orientation. For each:
    //    - set the appropriate orientation for the face or edge in what is otherwise an identity orientation for the cell
    //    - examine the data structure (the coefficient matrix) corresponding to that face or edge to determine whether it should be a permutation
    //    - place the orientation into a single-element Kokkos::View suitable for passing to OrientationTools::orientationsArePermutations()
    //    - verify that OrientationTools::orientationsArePermutations() returns the correct thing
    //    - store each cell orientation into one of two std::vectors, depending on whether it is a permutation or not.
    
    // 3. Place all permutation orientations into a single view; verify orientationsArePermutations() returns true
    
    // 4. Place all non-permutation orientations into a single view; verify orientationsArePermutations() returns false
    
    // 5. Place all permutation orientations and one non-permutation orientation (if available) into a single View; verify orientationsArePermutations() returns false
    
    std::vector<Orientation> permutationOrts, nonPermutationOrts;
    
    for (ordinal_type d=1; d<cellTopo.getDimension(); d++)
    {
      ordinal_type subcellCount = cellTopo.getSubcellCount(d);
      auto maxOrt = ortMax(cellTopo.getSubcell(d,0));
      
      for (ordinal_type scOrt=0; scOrt<maxOrt; scOrt++)
      {
        
      }
    }
    
    // orientationsArePermutations(const OrientationViewType orts, const BasisType * basis)
    
  }
  
  TEUCHOS_UNIT_TEST( OrientationTools, OrientationsArePermutations_Tet )
  {
    shards::CellTopology shardsTopo = shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() );
    Intrepid2::CellTopology cellTopo(shardsTopo, 0);
    
    int edgeDim = 1;
    int faceDim = 2;
    
//    cellTopo.getSubcell(<#int scdim#>, <#int scord#>)
  }
} // namespace
