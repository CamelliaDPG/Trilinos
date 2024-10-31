// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER


/** \file
    \brief  Comparing PAMatrix apply versus standard matrix assembly and apply().
    \author Nathan V. Roberts
*/

#include "Kokkos_Core.hpp"

#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_UnitTestRepository.hpp>

#include <Intrepid2_CellGeometryTestUtils.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Intrepid2_HGRAD_TET_Cn_FEM.hpp>
#include <Intrepid2_HGRAD_HEX_Cn_FEM.hpp>
#include <Intrepid2_IntegrationTools.hpp>
#include <Intrepid2_Kernels.hpp>
#include <Intrepid2_NodalBasisFamily.hpp>
#include <Intrepid2_TensorArgumentIterator.hpp>
#include <Intrepid2_TestUtils.hpp>

#include "Intrepid2_Data.hpp"
#include "Intrepid2_TensorData.hpp"
#include "Intrepid2_TensorPoints.hpp"
#include "Intrepid2_TransformedBasisValues.hpp"
#include "Intrepid2_VectorData.hpp"

#include "Intrepid2_ScalarView.hpp"
#include "PAMatrixAssembly.hpp"

#include "StructuredIntegrationTests_TagDefs.hpp"
#include "StructuredIntegrationTests_Utils.hpp"

namespace
{
using namespace Intrepid2;
template<typename DeviceType, typename Scalar>
using PAMatrix = ::Intrepid2::PAMatrix<DeviceType,Scalar>;

template<class Scalar, typename DeviceType>
void testPAMatrixApply(PAMatrix<DeviceType,Scalar> &paMatrix, const double &relTol, const double &absTol, Teuchos::FancyOStream &out, bool &success)
{
  // compare columns of the fully assembled matrix with the action of PAMatrix::apply() on unit vectors with a "1" in corresponding column position.
  
  auto fullMatrix = paMatrix.allocateMatrixStorage();
  paMatrix.assemble(fullMatrix);
  
  auto fullMatrixView = fullMatrix.getUnderlyingView(); // I believe this will always have shape (C,F1,F2) for our present inputs
  TEUCHOS_TEST_FOR_EXCEPTION(fullMatrixView.rank() != 3, std::invalid_argument, "underlying view does not have the expected rank");
  
  // construct a unit vector to extract a corresponding column
  const int numCells = fullMatrix.extent_int(0);
  const int numRows  = fullMatrix.extent_int(1); // (C,F1,F2)
  const int numCols  = fullMatrix.extent_int(2); // (C,F1,F2)
  
  ScalarView<Scalar, DeviceType>  inputVector( "inputVector",numCells,numCols); // C,F2
  ScalarView<Scalar, DeviceType> outputVector("outputVector",numCells,numRows); // C,F1
  
  Kokkos::deep_copy(inputVector,0.0);
  
  auto workspace1 = paMatrix.allocateWorkspace(numCells);
  auto workspace2 = paMatrix.allocateWorkspace(numCells);
  
  for (int colOrdinal=0; colOrdinal<numCols; colOrdinal++)
  {
    auto colSubView = Kokkos::subview(inputVector, Kokkos::ALL(), colOrdinal);
    Kokkos::deep_copy(colSubView, 1.0);
    
    paMatrix.apply(outputVector, inputVector, workspace1, workspace2);
  
    auto fullMatrixColumn = Kokkos::subview(fullMatrixView, Kokkos::ALL(), Kokkos::ALL(), colOrdinal);
    
    out << "checking col: " << colOrdinal << std::endl;
    testFloatingEquality2(fullMatrixColumn, outputVector, relTol, absTol, out, success, "fullMatrixColumn", "outputVector");
    
    // zero out the column entry we changed for next pass through the loop
    Kokkos::deep_copy(colSubView, 0.0);
  }
}

template<class Scalar, typename DeviceType>
void testSyntheticPAMatrixApply(std::vector< std::vector< std::vector< Scalar > > > basis1Ops,    // (component ordinal, F, P)
                                             std::vector< std::vector< Scalar > >   pointWeights, // (cell, P)
                                std::vector< std::vector< std::vector< Scalar > > > basis2Ops,    // (component ordinal, F, P)
                                const double &relTol, const double &absTol, Teuchos::FancyOStream &out, bool &success)
{
  using View2D = Kokkos::View<Scalar**,DeviceType>;
  std::vector<View2D> basis1OpViews;
  for (auto basis1Op : basis1Ops)
  {
    const int F = int(basis1Op.size());
    const int P = int(basis1Op[0].size());
    View2D basis1OpView("PAMatrixTests_Apply: basis1OpView",F,P);
    auto basis1OpViewHost = Kokkos::create_mirror_view(basis1OpView);
    for (int f=0; f<F; f++)
    {
      auto row = basis1Op[f];
      INTREPID2_TEST_FOR_EXCEPTION(row.size() != P, std::invalid_argument, "Each row of op container must have the same length");
      for (int p=0; p<P; p++)
      {
        basis1OpViewHost(f,p) = row[p];
      }
    }
    Kokkos::deep_copy(basis1OpView,basis1OpViewHost);
    basis1OpViews.push_back(basis1OpView);
  }
  std::vector<View2D> basis2OpViews;
  for (auto basis2Op : basis2Ops)
  {
    const int F = int(basis2Op.size());
    const int P = int(basis2Op[0].size());
    View2D basis2OpView("PAMatrixTests_Apply: basis2OpView",F,P);
    auto basis2OpViewHost = Kokkos::create_mirror_view(basis2OpView);
    for (int f=0; f<F; f++)
    {
      auto row = basis2Op[f];
      INTREPID2_TEST_FOR_EXCEPTION(row.size() != P, std::invalid_argument, "Each row of op container must have the same length");
      for (int p=0; p<P; p++)
      {
        basis2OpViewHost(f,p) = row[p];
      }
    }
    Kokkos::deep_copy(basis2OpView,basis2OpViewHost);
    basis2OpViews.push_back(basis2OpView);
  }
  const int C = int(pointWeights.size());
  const int P = int(pointWeights[0].size());
  View2D pointWeightsView("PAMatrixTests_Apply: pointWeightsView",C,P);
  auto pointWeightsViewHost = Kokkos::create_mirror_view(pointWeightsView);
  for (int c=0; c<C; c++)
  {
    auto row = pointWeights[c];
    INTREPID2_TEST_FOR_EXCEPTION(row.size() != P, std::invalid_argument, "Each row of pointWeights container must have the same length");
    for (int p=0; p<P; p++)
    {
      pointWeightsViewHost(c,p) = row[p];
    }
  }
  Kokkos::deep_copy(pointWeightsView,pointWeightsViewHost);
  auto paMatrix = syntheticPAMatrix<DeviceType,Scalar>(basis1OpViews, basis2OpViews, pointWeightsView);;
  
  testPAMatrixApply<Scalar, DeviceType>(paMatrix, relTol, absTol, out, success);
}

// MARK: ApplySynthetic1
TEUCHOS_UNIT_TEST(PAMatrix, ApplySynthetic1)
{
  // maybe the simplest possible test: put a single 1 entry in each container.
  using Scalar = double;
  double relTol = 1e-15;
  double absTol = 1e-15;
  
  std::vector< std::vector< std::vector< Scalar > > > basis1Ops {{{1.}}}, basis2Ops {{{1.}}};
  std::vector< std::vector< Scalar > > pointWeights {{1.}};
  testSyntheticPAMatrixApply<Scalar, DefaultTestDeviceType>(basis1Ops, pointWeights, basis2Ops, relTol, absTol, out, success);
}

} // anonymous namespace
