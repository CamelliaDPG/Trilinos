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
      case shards::Triangle<3>::key:      return 5;
      case shards::Quadrilateral<4>::key: return 7;
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
    
    std::vector<Orientation> permutationOrts, nonPermutationOrts;
    
    permutationOrts.push_back(Orientation()); // place identity there to start
    
    using BasisType = Basis<DeviceType,Scalar,Scalar>;
    const auto matData = OrientationTools<DeviceType>::createCoeffMatrix<BasisType>(basis.get());
    auto matDataHost = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), matData);
    
    ordinal_type edgeCount = cellTopo.getEdgeCount();
    
    for (ordinal_type d=1; d<cellTopo.getDimension(); d++)
    {
      ordinal_type subcellCount = cellTopo.getSubcellCount(d);
      auto maxOrt = ortMax(cellTopo.getSubcell(d,0));
      
      ScalarView<Orientation, DeviceType> singleOrtView("single-orientation view",1);
      
      for (ordinal_type sc=0; sc<subcellCount; sc++)
      {
        ordinal_type numDofs   = basis->getDofCount(d, sc);
        
        std::vector<ordinal_type> subcellOrts(subcellCount,0);
        // matData indices: (scIndex,scOrt,dofRow,dofCol)
        int scIndex = (d==1) ? sc : sc + edgeCount; // 2D faces get entries offset by edgeCount; no support for higher dimensions
        for (ordinal_type scOrt=1; scOrt<=maxOrt; scOrt++) // start with ort=1; ort=0 is the identity
        {
          Orientation ort; // starts as identity
          subcellOrts[sc] = scOrt;
          
          if (d == 1)
          {
            ort.setEdgeOrientation(subcellCount, &subcellOrts[0]);
          }
          else
          {
            ort.setFaceOrientation(subcellCount, &subcellOrts[0]);
          }
          
          bool isPermutation = true;
          // matData indices: (scIndex,scOrt,dofRow,dofCol)
          for (ordinal_type row=0; row<numDofs; row++)
          {
            int nnz = 0;
            for (ordinal_type col=0; col<numDofs; col++)
            {
              if (matDataHost(scIndex,scOrt,row,col) != 0)
              {
                nnz++;
              }
            }
            if (nnz != 1) isPermutation = false;
          }
          for (ordinal_type col=0; col<numDofs; col++)
          {
            int nnz = 0;
            for (ordinal_type row=0; row<numDofs; row++)
            {
              if (matDataHost(scIndex,scOrt,row,col) != 0)
              {
                nnz++;
              }
            }
            if (nnz != 1) isPermutation = false;
          }
          if (isPermutation)    permutationOrts.push_back(ort);
          else               nonPermutationOrts.push_back(ort);
          Kokkos::deep_copy(singleOrtView, ort);
          bool ortIsPermutation = OrientationTools<DeviceType>::orientationsArePermutations(singleOrtView, basis.get());
          TEST_EQUALITY(isPermutation, ortIsPermutation);
        }
      }
    }
    
    // 3. Place all permutation orientations into a single view; verify orientationsArePermutations() returns true
    // (there must always be at least one permutation orientation: the identity is one.)
    const ordinal_type numPermutationOrts = static_cast<ordinal_type>(permutationOrts.size());
    ScalarView<Orientation, DeviceType> permutationOrtsView("permutation orientations", numPermutationOrts);
    auto permutationOrtsViewHost = Kokkos::create_mirror(permutationOrtsView);
    for (ordinal_type permOrtOrdinal=0; permOrtOrdinal<numPermutationOrts; permOrtOrdinal++)
    {
      permutationOrtsViewHost(permOrtOrdinal) = permutationOrts[permOrtOrdinal];
    }
    Kokkos::deep_copy(permutationOrtsView, permutationOrtsViewHost);
    bool permutationOrtsArePermutations = OrientationTools<DeviceType>::orientationsArePermutations(permutationOrtsView, basis.get());
    TEST_EQUALITY(true, permutationOrtsArePermutations);
    
    // 4. Place all non-permutation orientations (if any) into a single view; verify orientationsArePermutations() returns false
    const ordinal_type numNonPermutationOrts = static_cast<ordinal_type>(nonPermutationOrts.size());
    if (numNonPermutationOrts > 0)
    {
      ScalarView<Orientation, DeviceType> nonPermutationOrtsView("non-permutation orientations", numNonPermutationOrts);
      auto nonPermutationOrtsViewHost = Kokkos::create_mirror(nonPermutationOrtsView);
      for (ordinal_type nonPermOrtOrdinal=0; nonPermOrtOrdinal<numNonPermutationOrts; nonPermOrtOrdinal++)
      {
        nonPermutationOrtsViewHost(nonPermOrtOrdinal) = nonPermutationOrts[nonPermOrtOrdinal];
      }
      Kokkos::deep_copy(nonPermutationOrtsView, nonPermutationOrtsViewHost);
      bool nonPermutationOrtsArePermutations = OrientationTools<DeviceType>::orientationsArePermutations(nonPermutationOrtsView, basis.get());
      TEST_EQUALITY(false, nonPermutationOrtsArePermutations);
    }
    
    // 5. Place all permutation orientations and one non-permutation orientation (if available) into a single View; verify orientationsArePermutations() returns false
    
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
