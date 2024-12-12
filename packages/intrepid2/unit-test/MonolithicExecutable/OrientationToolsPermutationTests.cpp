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

#include "Intrepid2_CellData.hpp"
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

/*** Tags for templated tests **/
class Tet
{
public:
  static const unsigned shardsTopoKey = shards::Tetrahedron<>::key;
};

class P1
{
public:
  static const int polyOrder = 1;
};
class P2
{
public:
  static const int polyOrder = 2;
};
class P3
{
public:
  static const int polyOrder = 3;
};
class P4
{
public:
  static const int polyOrder = 4;
};
class P5
{
public:
  static const int polyOrder = 5;
};

class HGRAD
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HGRAD;
};
class HDIV
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HDIV;
};
class HCURL
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HCURL;
};
class HVOL
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HVOL;
};

class Nodal
{
public:
  using BasisFamily = NodalBasisFamily<DeviceType,Scalar,Scalar>;
};
class DNodal
{
public:
  using BasisFamily = DerivedNodalBasisFamily<DeviceType,Scalar,Scalar>;
};
class Hierarchical
{
public:
  using BasisFamily = HierarchicalBasisFamily<DeviceType,Scalar,Scalar>;
};

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
    Orientation identityOrt; // default constructor for Orientation is identity
    ScalarView<Orientation, DeviceType> identityOrtView("identity orientations", 15);
    Kokkos::deep_copy(identityOrtView, identityOrt);
    
    bool identityIsPermutation = OrientationTools<DeviceType>::orientationsArePermutations(identityOrtView, basis.get());
    TEST_EQUALITY(identityIsPermutation, true);
    
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
     
    //    cellTopo.getSubcell(<#int scdim#>, <#int scord#>)
  }
  
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(OrientationTools, OrientationsArePermutations, TopoTag, FSTag, BasisFamilyTag, PolyOrderTag)
  {
    using BasisFamily = typename BasisFamilyTag::BasisFamily;
    
    using DataScalar  = double;
    using PointScalar = double;
    
    const unsigned      shardsKey = TopoTag::shardsTopoKey;
    const EFunctionSpace       fs =   FSTag::functionSpace;
    const ordinal_type  polyOrder = PolyOrderTag::polyOrder;

    shards::CellTopology shardsTopo(getCellTopologyData(shardsKey) );
    Intrepid2::CellTopology cellTopo(shardsTopo, 0);
    auto basis = getBasis<BasisFamily>(shardsTopo, fs, polyOrder);
    
    testOrientationsArePermutations(cellTopo, basis, out, success);
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(OrientationTools, OrientationsArePermutations, Tet, HCURL, Nodal, P4);
} // namespace
