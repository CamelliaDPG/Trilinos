// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Mauro Perego  (mperego@sandia.gov), or
//                    Nate Roberts  (nvrober@sandia.gov)
//
// ************************************************************************
// @HEADER


/** \file
    \brief  Test for checking accuracy of interpolation-based projections for pyramid elements, for exactly-representable functions

    The test considers a structured pyramid mesh of the cube [-1,1]^3, formed by first
    building a hexahedral mesh with N^3 hexes and then splitting each hex into 6 pyramids meeting
    in the center of the hexahedron.
    The test checks the accuracy of the HGRAD, HCURL, HDIV, HVOL projections of analytic
    target functions for Hierarchical basis functions as N increases.
    The accuracy is computed in the H^1, H^{curl}, H^{div} and L^2 norms respectively.
    The optimal order of convergence equates the basis degree.

    \author Created by Nate Roberts, based on convergence tests by Mauro Perego
 */

#include "Intrepid2_config.h"

#ifdef HAVE_INTREPID2_DEBUG
#define INTREPID2_TEST_FOR_DEBUG_ABORT_OVERRIDE_TO_CONTINUE
#endif

#include "Intrepid2_CellGeometry.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_ProjectionTools.hpp"
#include "Intrepid2_HGRAD_PYR_C1_FEM.hpp"
#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "struct_mesh_utils.hpp"

#define Intrepid2_Experimental


#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include <array>
#include <set>
#include <random>
#include <algorithm>

namespace Intrepid2 {

namespace Test {

#define INTREPID2_TEST_ERROR_EXPECTED( S )              \
    try {                                                               \
      ++nthrow;                                                         \
      S ;                                                               \
    }                                                                   \
    catch (std::exception &err) {                                        \
      ++ncatch;                                                         \
      *outStream << "Expected Error ----------------------------------------------------------------\n"; \
      *outStream << err.what() << '\n';                                 \
      *outStream << "-------------------------------------------------------------------------------" << "\n\n"; \
    }

template<typename ValueType, typename DeviceType>
int PatchProjectionPyr(const bool verbose) {

  using ExecSpaceType = typename DeviceType::execution_space;

  typedef Kokkos::DynRankView<ValueType,DeviceType> DynRankView;
  typedef Kokkos::DynRankView<ordinal_type,DeviceType> DynRankViewInt;
#define ConstructWithLabel(obj, ...) obj(#obj, __VA_ARGS__)

  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  if (verbose)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs,       false);

  Teuchos::oblackholestream oldFormatState;
  oldFormatState.copyfmt(std::cout);

  using HostSpaceType = Kokkos::DefaultHostExecutionSpace;

  *outStream << "DeviceSpace::  ";   ExecSpaceType().print_configuration(*outStream, false);
  *outStream << "HostSpace::    ";   HostSpaceType().print_configuration(*outStream, false);
  *outStream << "\n";

  int errorFlag = 0;
  const ValueType tol = 1e-12;

  struct Fun {
    const ordinal_type degree;
    
    KOKKOS_INLINE_FUNCTION
    ValueType
    operator()(const ValueType& x, const ValueType& y, const ValueType& z) {
      return std::pow(x+y-z,degree-1)*(x+y-2);
    }
  };

  struct GradFun {
    const ordinal_type degree;
    
    KOKKOS_INLINE_FUNCTION
    ValueType
    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
      switch (comp) {
      case 0:
        return  (degree-1) * std::pow(x+y-z,degree-2)*(x+y-2) + std::pow(x+y-z,degree-1);
      case 1:
        return  (degree-1) * std::pow(x+y-z,degree-2)*(x+y-2) + std::pow(x+y-z,degree-1);
      case 2:
        return -(degree-1) * std::pow(x+y-z,degree-2)*(x+y-2);
      default:
        return 0;
      }
    }
  };

  struct BasisFunCurl {
    const ordinal_type degree;
    const ordinal_type dofOrdinal;
    const EOperator op;
    
    HierarchicalBasis_HCURL_PYR<DeviceType,ValueType,ValueType> hcurlPyrBasis;
    
    DynRankView singleValueView;
    DynRankView singlePointView;
    
    BasisFunCurl(const ordinal_type polyOrder, const ordinal_type dof, const EOperator operatorType)
    :
    degree(polyOrder),
    dofOrdinal(dof),
    op(operatorType),
    hcurlPyrBasis(polyOrder)
    {
      const ordinal_type basisCardinality = hcurlPyrBasis.getCardinality();
      const ordinal_type numPoints = 1;
      const ordinal_type spaceDim = 3;
      singleValueView = DynRankView("singleValueView",basisCardinality,numPoints,spaceDim);
      singlePointView = DynRankView("singlePointView",numPoints,spaceDim);
    }
    
    KOKKOS_INLINE_FUNCTION
    ValueType
    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
      singlePointView(0,0) = x;
      singlePointView(0,1) = y;
      singlePointView(0,2) = z;
      
      // note that the following call is *not* safe for calling on device (it invokes a device-level parallel_for)
      hcurlPyrBasis.getValues(singleValueView, singlePointView, op);
      
      //fun = f + a
      switch (comp) {
        case 0:
        case 1:
        case 2:
          return singleValueView(dofOrdinal,0,comp);
        default:
          return 0;
      }
    }
  };
  
//  struct FunCurl {
//    const ordinal_type degree;
//
//    KOKKOS_INLINE_FUNCTION
//    ValueType
//    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
//      ValueType a0 = std::pow(x-y+z, degree-1);
//      ValueType a1 = std::pow(2-y+z, degree-1);
//      ValueType a2 = std::pow(x-1, degree-1);
//      ValueType f0 = 3;
//      ValueType f1 = std::pow(y, degree-1);
//      ValueType f2 = std::pow(x+z, degree-1);
//      //fun = f + a
//      switch (comp) {
//      case 0:
//        return f0 + a0;
//      case 1:
//        return f1 + a1;
//      case 2:
//        return f2 + a2;
//      default:
//        return 0;
//      }
//    }
//  };
//
//  struct CurlFunCurl {
//    const ordinal_type degree;
//
//    KOKKOS_INLINE_FUNCTION
//    ValueType
//    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
//      ValueType df0_dx = 0;
//      ValueType df0_dy = 0;
//      ValueType df0_dz = 0;
//
//      ValueType df1_dx = 0;
//      ValueType df1_dy = (degree-1) * std::pow(y, degree-2);
//      ValueType df1_dz = 0;
//
//      const bool xpzIsZero = std::abs(x+z) < 1e-14;
//      ValueType df2_dx = xpzIsZero ? 0 : (degree-1) * std::pow(x+z, degree-2);
//      ValueType df2_dy = xpzIsZero ? 0 : 0;
//      ValueType df2_dz = xpzIsZero ? 0 : (degree-1) * std::pow(x+z, degree-2);
//
//      const bool xmypzIsZero = std::abs(x+z) < 1e-14;
//      ValueType da0_dx = xmypzIsZero ? 0 :   (degree-1) * std::pow(x-y+z, degree-2);
//      ValueType da0_dy = xmypzIsZero ? 0 : - (degree-1) * std::pow(x-y+z, degree-2);
//      ValueType da0_dz = xmypzIsZero ? 0 :   (degree-1) * std::pow(x-y+z, degree-2);
//
//      const bool two_minus_y_plus_z_IsZero = std::abs(2-y+z) < 1e-14;
//      ValueType da1_dx = two_minus_y_plus_z_IsZero ? 0 : 0;
//      ValueType da1_dy = two_minus_y_plus_z_IsZero ? 0 : - (degree-1) * std::pow(2-y+z, degree-2);
//      ValueType da1_dz = two_minus_y_plus_z_IsZero ? 0 :   (degree-1) * std::pow(2-y+z, degree-2);
//
//      ValueType da2_dx = (degree-1) * std::pow(x-1, degree-2);
//      ValueType da2_dy = 0;
//      ValueType da2_dz = 0;
//
//      switch (comp) {
//        case 0:
//          return df2_dy - df1_dz + da2_dy - da1_dz;
//        case 1:
//          return df0_dz - df2_dx + da0_dz - da2_dx;
//        case 2:
//          return df1_dx - df0_dy + da1_dx - da0_dy;
//      default:
//        return 0;
//      }
//    }
//  };
  
  
  struct FunDiv {
    const ordinal_type degree;
    
    KOKKOS_INLINE_FUNCTION
    ValueType
    operator()(const ValueType& x, const ValueType& y, const ValueType& z, const int comp=0) {
      ValueType a0 = std::pow(x-y+z, degree-1);
      ValueType a1 = std::pow(2-y+z, degree-1);
      ValueType a2 = std::pow(x-1, degree-1);
      ValueType f0 = 3;
      ValueType f1 = std::pow(y, degree-1);
      ValueType f2 = std::pow(x+z, degree-1);
      //fun = f + a
      switch (comp) {
        case 0:
          return f0 + a0;
        case 1:
          return f1 + a1;
        case 2:
          return f2 + a2;
        default:
          return 0;
      }
    }
  };

  struct DivFunDiv {
    const ordinal_type degree;
    
    KOKKOS_INLINE_FUNCTION
    ValueType
    operator()(const ValueType& x, const ValueType& y, const ValueType& z) {
      const bool yIsZero     = std::abs(y)     < 1e-14;
      const bool xpzIsZero   = std::abs(x+z)   < 1e-14;
      const bool xmypzIsZero = std::abs(x-y+z) < 1e-14;
      const bool zmyp2IsZero = std::abs(z-y+2) < 1e-14;
      
      ValueType df0_dx = 0;
      ValueType df1_dy =   yIsZero ? 0 : (degree-1) * std::pow(  y, degree-2);
      ValueType df2_dz = xpzIsZero ? 0 : (degree-1) * std::pow(x+z, degree-2);
      
      ValueType da0_dx = xmypzIsZero ? 0 :   (degree-1) * std::pow(x-y+z, degree-2);
      ValueType da1_dy = zmyp2IsZero ? 0 : - (degree-1) * std::pow(2-y+z, degree-2);
      ValueType da2_dz = 0;
      
      return df0_dx + df1_dy + df2_dz + da0_dx + da1_dy + da2_dz;
    }
  };
  
  typedef CellTools<DeviceType> ct;
  typedef OrientationTools<DeviceType> ots;
  typedef Experimental::ProjectionTools<DeviceType> pts;
  typedef RealSpaceTools<DeviceType> rst;
  typedef FunctionSpaceTools<DeviceType> fst;

  constexpr ordinal_type dim = 3;
//  const ordinal_type maxBasisDegree = 9;
  ordinal_type cub_degree = 9;
  const ordinal_type basisDegree = 1;

  // ************************************ GET INPUTS **************************************

  int NX = 2;
  constexpr int numRefinements = 2; // change to 4 to reproduce the full set of values below.

  // Expected values of the projection errors in H1, Hcurl, Hdiv and L2 norms for HGRAD, HDIV, HCURL and HVOL elements respectively.
  // These values have been computed running the code with numRefinements=4 and the convergence rates are close to the optimal ones.
  // Note that these values are independent of the basis choice (Hierarchical or Nodal) as long as they generate the same functional space.
  // We currently only test two mesh refinements to make the test run faster, so this is used as a regression test rather than
  // a convergence test, but the test can be used for verifying optimal accuracy as well.
  ValueType hgradNorm[numRefinements];
  ValueType hcurlNorm[numRefinements];
  ValueType hdivNorm[numRefinements];
  ValueType hvolNorm[numRefinements];

  ValueType hgrad_errors[4] = {0, 0, 0, 0};
  ValueType hcurl_errors[4] = {0, 0, 0, 0};
  ValueType hdiv_errors[4] = {0, 0, 0, 0};
  ValueType hvol_errors[4] = {0, 0, 0, 0};

  ValueType hgrad_errors_L2[4] = {0, 0, 0, 0};
  ValueType hcurl_errors_L2[4] = {0, 0, 0, 0};
  ValueType hdiv_errors_L2[4] = {0, 0, 0, 0};
  ValueType hvol_errors_L2[4] = {0, 0, 0, 0};

  for(int iter= 0; iter<numRefinements; iter++, NX *= 2) {
    int NY            = NX;
    int NZ            = NX;

    // *********************************** CELL TOPOLOGY **********************************

    // Get cell topology for base tetrahedron
    typedef shards::CellTopology    CellTopology;
    CellTopology cellTopo(shards::getCellTopologyData<shards::Pyramid<5> >() );

    // Get dimensions
    ordinal_type numNodesPerElem = cellTopo.getNodeCount();

    // *********************************** GENERATE MESH ************************************
    
    Kokkos::Array<ValueType,dim> origin{-1,-1,-1};
    Kokkos::Array<ValueType,dim> domainExtents{2,2,2};
    Kokkos::Array<int,dim> gridCellCounts{NX,NY,NZ};
    
    using CellGeometryType = CellGeometry<ValueType, dim, DeviceType>;
    
//    nodes - (GN,D) container specifying the coordinate weight for the global node in the specified dimension; if cellToNodes is not allocated, this must be a (C,N,D) container
    
    ordinal_type numElems = 1;
    DynRankViewInt emptyCellToNodes;
    
//    vertex ordering: (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1)
    
    DynRankView ConstructWithLabel(physVertexes, numElems, numNodesPerElem, dim);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
    KOKKOS_LAMBDA (const int &i) {
//      physVertexes(i,0,0) = 0;
//      physVertexes(i,0,1) = 0;
//      physVertexes(i,0,2) = 0;
//
//      physVertexes(i,1,0) = 1;
//      physVertexes(i,1,1) = 0;
//      physVertexes(i,1,2) = 0;
//
//      physVertexes(i,2,0) = 1;
//      physVertexes(i,2,1) = 1;
//      physVertexes(i,2,2) = 0;
//
//      physVertexes(i,3,0) = 0;
//      physVertexes(i,3,1) = 1;
//      physVertexes(i,3,2) = 0;
//
//      physVertexes(i,4,0) = 0;
//      physVertexes(i,4,1) = 0;
//      physVertexes(i,4,2) = 1;
      
      physVertexes(i,0,0) = -1;
      physVertexes(i,0,1) = -1;
      physVertexes(i,0,2) = 0;
      
      physVertexes(i,1,0) = 1;
      physVertexes(i,1,1) = -1;
      physVertexes(i,1,2) = 0;
      
      physVertexes(i,2,0) = 1;
      physVertexes(i,2,1) = 1;
      physVertexes(i,2,2) = 0;
      
      physVertexes(i,3,0) = -1;
      physVertexes(i,3,1) = 1;
      physVertexes(i,3,2) = 0;
      
      physVertexes(i,4,0) = 0;
      physVertexes(i,4,1) = 0;
      physVertexes(i,4,2) = 1;
    });
    
    CellGeometryType cellGeometry(cellTopo,emptyCellToNodes, physVertexes);
    
//    CellGeometryType cellGeometry(origin, domainExtents, gridCellCounts, CellGeometryType::SIX_PYRAMIDS);

//    //computing vertices coords
//    DynRankView ConstructWithLabel(physVertexes, numElems, numNodesPerElem, dim);
//    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
//    KOKKOS_LAMBDA (const int &i) {
//      for(ordinal_type j=0; j<numNodesPerElem; ++j)
//        for(ordinal_type k=0; k<dim; ++k)
//        {
//          physVertexes(i,j,k) = cellGeometry(i,j,k);
//        }
//    });
//    ExecSpaceType().fence();

    DefaultCubatureFactory cub_factory;
    auto cell_cub = cub_factory.create<DeviceType, ValueType, ValueType>(cellTopo.getBaseKey(), cub_degree);
    ordinal_type numRefCoords = cell_cub->getNumPoints();
    DynRankView ConstructWithLabel(refPoints, numRefCoords, dim);
    DynRankView ConstructWithLabel(weights, numRefCoords);
    cell_cub->getCubature(refPoints, weights);

    using basisType = Basis<DeviceType,ValueType,ValueType>;
    using CG_HBasis = HierarchicalBasisFamily<DeviceType,ValueType,ValueType>;

    std::vector<bool> useL2Proj;
    useL2Proj.push_back(true); // use L2 projection for all the bases
    useL2Proj.push_back(false);

    bool testHGRAD = false;
    
    if (testHGRAD)
    {
      *outStream
      << "===============================================================================\n"
      << "|                                                                             |\n"
      << "|                 Test 1 (Patch Projection - HGRAD)                           |\n"
      << "|                                                                             |\n"
      << "===============================================================================\n";

      try {
        // compute orientations for cells (one-time computation)
        Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numElems);
        cellGeometry.orientations(elemOrts);
        
        for (auto useL2Projection:useL2Proj) { //

        std::vector<basisType*> basis_set;
        basis_set.push_back(new typename  CG_HBasis::HGRAD_PYR(basisDegree));

        for (auto basisPtr:basis_set) {
          auto& basis = *basisPtr;
          *outStream << " " << basis.getName() << std::endl;
          ordinal_type basisCardinality = basis.getCardinality();

          //Compute Reference coordinates
          DynRankView ConstructWithLabel(physRefCoords, numElems, numRefCoords, dim);
          {
            Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> linearBasis; //used for computing physical coordinates
            DynRankView ConstructWithLabel(linearBasisValuesAtRefCoords, numNodesPerElem, numRefCoords);
            linearBasis.getValues(linearBasisValuesAtRefCoords, refPoints);
            ExecSpaceType().fence();
            Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &i) {
              for(ordinal_type d=0; d<dim; ++d)
                for(ordinal_type j=0; j<numRefCoords; ++j)
                  for(ordinal_type k=0; k<numNodesPerElem; ++k)
                    physRefCoords(i,j,d) += cellGeometry(i,k,d)*linearBasisValuesAtRefCoords(k,j);
            });
            ExecSpaceType().fence();
          }

          DynRankView ConstructWithLabel(funAtRefCoords, numElems, numRefCoords);
          DynRankView ConstructWithLabel(funGradAtPhysRefCoords, numElems, numRefCoords, dim);

          Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
          KOKKOS_LAMBDA (const int &i) {
            Fun fun {basisDegree};
            GradFun gradFun {basisDegree};
            for(ordinal_type j=0; j<numRefCoords; ++j) {
              funAtRefCoords(i,j) = fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2));
              for(ordinal_type d=0; d<dim; ++d)
                funGradAtPhysRefCoords(i,j,d) = gradFun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2),d);
            }
          });
          ExecSpaceType().fence();

          // compute projection-based interpolation of fun into HGRAD
          DynRankView ConstructWithLabel(basisCoeffsHGrad, numElems, basisCardinality);
          {
            ordinal_type targetCubDegree(basis.getDegree()),targetDerivCubDegree(basis.getDegree());

            Experimental::ProjectionStruct<DeviceType,ValueType> projStruct;
            if(useL2Projection) {
              projStruct.createL2ProjectionStruct(&basis, targetCubDegree);
            } else {
              projStruct.createHGradProjectionStruct(&basis, targetCubDegree, targetDerivCubDegree);
            }
            
            auto evaluationPoints = projStruct.getAllEvalPoints();
            auto evaluationGradPoints = projStruct.getAllDerivEvalPoints();
            ordinal_type numPoints = evaluationPoints.extent(0), numGradPoints = evaluationGradPoints.extent(0);

            DynRankView ConstructWithLabel(targetAtEvalPoints, numElems, numPoints);
            DynRankView ConstructWithLabel(targetGradAtEvalPoints, numElems, numGradPoints, dim);

            DynRankView ConstructWithLabel(physEvalPoints, numElems, numPoints, dim);
            DynRankView ConstructWithLabel(physEvalGradPoints, numElems, numGradPoints, dim);
            {
              DynRankView ConstructWithLabel(linearBasisValuesAtEvalPoint, numElems, numNodesPerElem);
              DynRankView ConstructWithLabel(linearBasisValuesAtEvalGradPoint, numElems, numNodesPerElem);

              Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
              KOKKOS_LAMBDA (const int &i) {
                auto basisValuesAtEvalPoint = Kokkos::subview(linearBasisValuesAtEvalPoint,i,Kokkos::ALL());
                for(ordinal_type j=0; j<numPoints; ++j){
                  auto evalPoint = Kokkos::subview(evaluationPoints,j,Kokkos::ALL());
                  Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalPoint, evalPoint);
                  for(ordinal_type k=0; k<numNodesPerElem; ++k)
                    for(ordinal_type d=0; d<dim; ++d)
                      physEvalPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalPoint(k);
                }

                auto basisValuesAtEvalGradPoint = Kokkos::subview(linearBasisValuesAtEvalGradPoint,i,Kokkos::ALL());
                for(ordinal_type j=0; j<numGradPoints; ++j) {
                  auto evalGradPoint = Kokkos::subview(evaluationGradPoints,j,Kokkos::ALL());
                  Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalGradPoint, evalGradPoint);
                  for(ordinal_type k=0; k<numNodesPerElem; ++k)
                    for(ordinal_type d=0; d<dim; ++d)
                      physEvalGradPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalGradPoint(k);
                }
              });
              ExecSpaceType().fence();
            }

            //transform the target function and its derivative to the reference element (inverse of pullback operator)
            DynRankView ConstructWithLabel(jacobian, numElems, numGradPoints, dim, dim);
            if(numGradPoints>0)
              ct::setJacobian(jacobian, evaluationGradPoints, physVertexes, cellTopo);
            
            Kokkos::deep_copy(targetGradAtEvalPoints,0.);
            Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &ic) {
              Fun fun {basisDegree};
              GradFun gradFun {basisDegree};
              for(int i=0;i<numPoints;i++) {
                targetAtEvalPoints(ic,i) = fun(physEvalPoints(ic,i,0), physEvalPoints(ic,i,1), physEvalPoints(ic,i,2));
              }
              for(int i=0;i<numGradPoints;i++) {
                for(int d=0;d<dim;d++)
                  for(int j=0;j<dim;j++)
                    targetGradAtEvalPoints(ic,i,j) += jacobian(ic,i,d,j)*gradFun(physEvalGradPoints(ic,i,0), physEvalGradPoints(ic,i,1), physEvalGradPoints(ic,i,2), d);//funHGradCoeffs(k)
              }
            });
            ExecSpaceType().fence();

            if(useL2Projection) {
              pts::getL2BasisCoeffs(basisCoeffsHGrad,
                  targetAtEvalPoints,
                  elemOrts,
                  &basis,
                  &projStruct);
            } else {
              pts::getHGradBasisCoeffs(basisCoeffsHGrad,
                  targetAtEvalPoints,
                  targetGradAtEvalPoints,
                  elemOrts,
                  &basis,
                  &projStruct);
            }
          }

          //check that fun values at reference points coincide with those computed using basis functions
          DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords);
          DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords);
          DynRankView basisValuesAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords);

          DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords);
          basis.getValues(basisValuesAtRefCoords, refPoints);
          rst::clone(basisValuesAtRefCoordsCells,basisValuesAtRefCoords);

          // modify basis values to account for orientations
          ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
              basisValuesAtRefCoordsCells,
              elemOrts,
              &basis);

          // transform basis values
          deep_copy(transformedBasisValuesAtRefCoordsOriented,
              basisValuesAtRefCoordsOriented);

          DynRankView ConstructWithLabel(basisGradsAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
          DynRankView ConstructWithLabel(transformedBasisGradsAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
          DynRankView basisGradsAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords, dim);

          DynRankView ConstructWithLabel(basisGradsAtRefCoords, basisCardinality, numRefCoords, dim);
          basis.getValues(basisGradsAtRefCoords, refPoints,OPERATOR_GRAD);
          rst::clone(basisGradsAtRefCoordsCells,basisGradsAtRefCoords);

          // modify basis values to account for orientations
          ots::modifyBasisByOrientation(basisGradsAtRefCoordsOriented,
              basisGradsAtRefCoordsCells,
              elemOrts,
              &basis);

          // transform basis values to the reference element (pullback)
          DynRankView ConstructWithLabel(jacobianAtRefCoords, numElems, numRefCoords, dim, dim);
          DynRankView ConstructWithLabel(jacobianAtRefCoords_inv, numElems, numRefCoords, dim, dim);
          DynRankView ConstructWithLabel(jacobianAtRefCoords_det, numElems, numRefCoords);
          ct::setJacobian(jacobianAtRefCoords, refPoints, physVertexes, cellTopo);
          ct::setJacobianInv (jacobianAtRefCoords_inv, jacobianAtRefCoords);
          ct::setJacobianDet (jacobianAtRefCoords_det, jacobianAtRefCoords);
          
  //        {
  //          for (int i=0; i<numElems; i++)
  //          {
  //            for (int j=0; j<numRefCoords; j++)
  //            {
  //              for (int d1=0; d1<dim; d1++)
  //              {
  //                for (int d2=0; d2<dim; d2++)
  //                {
  //                  std::cout << "jacobianAtRefCoords(" << i << "," << j << "," << d1 << "," << d2 << "): " << jacobianAtRefCoords(i,j,d1,d2) << std::endl;
  //                }
  //              }
  //            }
  //          }
  //          for (int i=0; i<numElems; i++)
  //          {
  //            for (int j=0; j<numRefCoords; j++)
  //            {
  //              for (int d1=0; d1<dim; d1++)
  //              {
  //                for (int d2=0; d2<dim; d2++)
  //                {
  //                  std::cout << "jacobianAtRefCoords_inv(" << i << "," << j << "," << d1 << "," << d2 << "): " << jacobianAtRefCoords_inv(i,j,d1,d2) << std::endl;
  //                }
  //              }
  //            }
  //          }
  //          for (int i=0; i<numElems; i++)
  //          {
  //            for (int j=0; j<numRefCoords; j++)
  //            {
  //              std::cout << "jacobianAtRefCoords_det(" << i << "," << j << "): " << jacobianAtRefCoords_det(i,j) << std::endl;
  //            }
  //          }
  //        }
          
          
          fst::HCURLtransformVALUE(transformedBasisGradsAtRefCoordsOriented,
              jacobianAtRefCoords_inv,
              basisGradsAtRefCoordsOriented);

          DynRankView ConstructWithLabel(projectedFunAtRefCoords, numElems, numRefCoords);
          DynRankView ConstructWithLabel(funGradAtRefCoordsOriented, numElems, numRefCoords,dim);

          //compute error of projection in H1 norm
          ValueType norm2(0);
          Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
          KOKKOS_LAMBDA (const int &i, double &norm2Update) {
            for(ordinal_type j=0; j<numRefCoords; ++j) {
              for(ordinal_type k=0; k<basisCardinality; ++k) {
                projectedFunAtRefCoords(i,j) += basisCoeffsHGrad(i,k)*transformedBasisValuesAtRefCoordsOriented(i,k,j);
                for (ordinal_type d=0; d<dim; ++d)
                  funGradAtRefCoordsOriented(i,j,d) += basisCoeffsHGrad(i,k)*transformedBasisGradsAtRefCoordsOriented(i,k,j,d);
              }
              {
  //              std::cout << "norm2Update = " << norm2Update << std::endl;
  //              // DEBUGGING
  //              std::cout << "funAtRefCoords("<< i << "," << j<< "): " << funAtRefCoords(i,j) << std::endl;
  //              std::cout << "projectedFunAtRefCoords(" << i << "," << j << "): " << projectedFunAtRefCoords(i,j) << std::endl;
  //              for (ordinal_type d=0; d<dim; ++d)
  //              {
  //                std::cout << "funGradAtPhysRefCoords(" << i << "," << j << "," << d << "): " << funGradAtPhysRefCoords(i,j,d) << std::endl;
  //                std::cout << "funGradAtRefCoordsOriented(" << i << "," << j << "," << d << "): " << funGradAtRefCoordsOriented(i,j,d) << std::endl;
  //              }
              }
              const auto absJacobianDet = (jacobianAtRefCoords_det(i,j) < 0) ? -jacobianAtRefCoords_det(i,j) : jacobianAtRefCoords_det(i,j);
              
              norm2Update += (funAtRefCoords(i,j) - projectedFunAtRefCoords(i,j))*
                  (funAtRefCoords(i,j) - projectedFunAtRefCoords(i,j))*
                  weights(j)*absJacobianDet;
              for (ordinal_type d=0; d<dim; ++d)
                norm2Update += (funGradAtPhysRefCoords(i,j,d) - funGradAtRefCoordsOriented(i,j,d))*
                  (funGradAtPhysRefCoords(i,j,d) - funGradAtRefCoordsOriented(i,j,d))*
                  weights(j)*absJacobianDet;
            }
          }, norm2);

          ExecSpaceType().fence();

          hgradNorm[iter] =  std::sqrt(norm2);
          auto expected_error = useL2Projection ? hgrad_errors_L2[iter] : hgrad_errors[iter];
          if (std::isnan(hgradNorm[iter]))
          {
            errorFlag++;
            *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
            *outStream << "For N = " << NX << ", computed error (" << hgradNorm[iter] << ") is nan!";
            *outStream << std::endl;
          }
          else if(std::abs(hgradNorm[iter]-expected_error) > tol){
            errorFlag++;
            *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
            *outStream << "For N = " << NX << ", computed error (" << hgradNorm[iter] << ") is different than expected one (" << expected_error << ")";
            *outStream << std::endl;
          }
          delete basisPtr;
        }
        if(useL2Projection)
          *outStream << "HGRAD Error (L2 Projection): " << hgradNorm[iter] <<std::endl;
        else
          *outStream << "HGRAD Error (HGrad Projection): " << hgradNorm[iter] <<std::endl;
        }
      } catch (std::exception &err) {
        std::cout << " Exception\n";
        *outStream << err.what() << "\n\n";
        errorFlag = -1000;
      }
    } // testHGRAD

    *outStream
    << "===============================================================================\n"
    << "|                                                                             |\n"
    << "|                 Test 2 (Patch Projection - HCURL)                           |\n"
    << "|                                                                             |\n"
    << "===============================================================================\n";


    try {
      // compute orientations for cells (one time computation)
      Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numElems);
      cellGeometry.orientations(elemOrts);

      for (auto useL2Projection:useL2Proj) { 
      
//      for (int p=1; p<9; p++)
      for (int p=2; p<3; p++)
      {
        std::vector<Teuchos::RCP<basisType>> basis_set;
        basis_set.push_back(Teuchos::rcp(new typename  CG_HBasis::HCURL_PYR(p)));

        for (auto basisPtr:basis_set) {
          auto& basis = *basisPtr;
          *outStream << " " << basis.getName() << std::endl;
          ordinal_type basisCardinality = basis.getCardinality();

          //Compute physical Dof Coordinates and Reference coordinates
          DynRankView ConstructWithLabel(physRefCoords, numElems, numRefCoords, dim);
          {
            Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> linearBasis; //used for computing physical coordinates
            DynRankView ConstructWithLabel(linearBasisValuesAtRefCoords, numNodesPerElem, numRefCoords);
            linearBasis.getValues(linearBasisValuesAtRefCoords, refPoints);
            ExecSpaceType().fence();
            Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &i) {
              for(ordinal_type d=0; d<dim; ++d)
                for(ordinal_type j=0; j<numRefCoords; ++j)
                  for(ordinal_type k=0; k<numNodesPerElem; ++k)
                    physRefCoords(i,j,d) += cellGeometry(i,k,d)*linearBasisValuesAtRefCoords(k,j);
            });
            ExecSpaceType().fence();
          }
          using std::cout;
          using std::endl;
          for (int i=0; i<numElems; i++)
          {
            for (int pt=0; pt<numRefCoords; pt++)
            {
              for (int d0=0; d0<dim; d0++)
              {
                cout << "physRefCoords(" << i <<"," << pt << "," << d0 << "): " <<  physRefCoords(i,pt,d0) << endl;
              }
            }
          }
          
          //check function reproducibility
          for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++)
//          for (int dofOrdinal=8; dofOrdinal<9; dofOrdinal++)
          {
            DynRankView ConstructWithLabel(    funAtPhysRefCoords, numElems, numRefCoords, dim);
            DynRankView ConstructWithLabel(funCurlAtPhysRefCoords, numElems, numRefCoords, dim);
            Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &i) {
              BasisFunCurl fun(p, dofOrdinal, OPERATOR_VALUE);
              BasisFunCurl curlFun(p, dofOrdinal, OPERATOR_CURL);
              for(ordinal_type j=0; j<numRefCoords; ++j) {
                for(ordinal_type k=0; k<dim; ++k) {
                      funAtPhysRefCoords(i,j,k) =     fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2), k);
                  funCurlAtPhysRefCoords(i,j,k) = curlFun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2), k);
                }
              }
            });
            ExecSpaceType().fence();

            // compute projection-based interpolation of fun into HCURL
            DynRankView ConstructWithLabel(basisCoeffsHCurl, numElems, basisCardinality);
            {
              ordinal_type targetCubDegree(cub_degree),targetDerivCubDegree(cub_degree-1);

              Experimental::ProjectionStruct<DeviceType,ValueType> projStruct;
              if(useL2Projection) {
                projStruct.createL2ProjectionStruct(&basis, targetCubDegree);
              } else {
                projStruct.createHCurlProjectionStruct(&basis, targetCubDegree, targetDerivCubDegree);
              }

              auto evaluationPoints = projStruct.getAllEvalPoints();
              auto evaluationCurlPoints = projStruct.getAllDerivEvalPoints();
              ordinal_type numPoints = evaluationPoints.extent(0), numCurlPoints = evaluationCurlPoints.extent(0);

              DynRankView ConstructWithLabel(targetAtEvalPoints, numElems, numPoints, dim);
              DynRankView ConstructWithLabel(targetCurlAtEvalPoints, numElems, numCurlPoints, dim);


              DynRankView ConstructWithLabel(physEvalPoints, numElems, numPoints, dim);
              DynRankView ConstructWithLabel(physEvalCurlPoints, numElems, numCurlPoints, dim);
              {
                DynRankView ConstructWithLabel(linearBasisValuesAtEvalPoint, numElems, numNodesPerElem);
                DynRankView ConstructWithLabel(linearBasisValuesAtEvalCurlPoint, numElems, numNodesPerElem);

                Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
                KOKKOS_LAMBDA (const int &i) {
                  auto basisValuesAtEvalPoint = Kokkos::subview(linearBasisValuesAtEvalPoint,i,Kokkos::ALL());
                  for(ordinal_type j=0; j<numPoints; ++j){
                    auto evalPoint = Kokkos::subview(evaluationPoints,j,Kokkos::ALL());
                    Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalPoint, evalPoint);
                    for(ordinal_type k=0; k<numNodesPerElem; ++k)
                      for(ordinal_type d=0; d<dim; ++d)
                        physEvalPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalPoint(k);
                  }

                  auto basisValuesAtEvalCurlPoint = Kokkos::subview(linearBasisValuesAtEvalCurlPoint,i,Kokkos::ALL());
                  for(ordinal_type j=0; j<numCurlPoints; ++j) {
                    auto evalGradPoint = Kokkos::subview(evaluationCurlPoints,j,Kokkos::ALL());
                    Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalCurlPoint, evalGradPoint);
                    for(ordinal_type k=0; k<numNodesPerElem; ++k)
                      for(ordinal_type d=0; d<dim; ++d)
                        physEvalCurlPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalCurlPoint(k);
                  }
                });
                ExecSpaceType().fence();
              }

              //transform the target function and its derivative to the reference element (inverse of pullback operator)
              DynRankView ConstructWithLabel(jacobian, numElems, numPoints, dim, dim);
              ct::setJacobian(jacobian, evaluationPoints, physVertexes, cellTopo);


              DynRankView ConstructWithLabel(jacobianCurl_inv, numElems, numCurlPoints, dim, dim);
              DynRankView ConstructWithLabel(jacobianCurl_det, numElems, numCurlPoints);
              if(numCurlPoints>0){
                DynRankView ConstructWithLabel(jacobianCurl, numElems, numCurlPoints, dim, dim);
                ct::setJacobian(jacobianCurl, evaluationCurlPoints, physVertexes, cellTopo);
                ct::setJacobianInv (jacobianCurl_inv, jacobianCurl);
                ct::setJacobianDet (jacobianCurl_det, jacobianCurl);
              }

              Kokkos::deep_copy(targetCurlAtEvalPoints,0.);
              Kokkos::deep_copy(targetAtEvalPoints,0.);
              Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
              KOKKOS_LAMBDA (const int &ic) {
                BasisFunCurl fun(p, dofOrdinal, OPERATOR_VALUE);
                BasisFunCurl curlFun(p, dofOrdinal, OPERATOR_CURL);
                for(int i=0;i<numPoints;i++) {
                  for(int j=0;j<dim;j++)
                    for(int d=0;d<dim;d++)
                      targetAtEvalPoints(ic,i,j) += jacobian(ic,i,d,j)*fun(physEvalPoints(ic,i,0), physEvalPoints(ic,i,1), physEvalPoints(ic,i,2),d);
                }
                for(int i=0;i<numCurlPoints;i++) {
                  for(int d=0;d<dim;d++)
                    for(int j=0;j<dim;j++)
                      targetCurlAtEvalPoints(ic,i,j) += jacobianCurl_det(ic,i)*jacobianCurl_inv(ic,i,j,d)*curlFun(physEvalCurlPoints(ic,i,0), physEvalCurlPoints(ic,i,1), physEvalCurlPoints(ic,i,2), d);//funHGradCoeffs(k)
                }
              });
              ExecSpaceType().fence();

              if(useL2Projection) {
                pts::getL2BasisCoeffs(basisCoeffsHCurl,
                    targetAtEvalPoints,
                    elemOrts,
                    &basis,
                    &projStruct);
              } else {
                pts::getHCurlBasisCoeffs(basisCoeffsHCurl,
                    targetAtEvalPoints,
                    targetCurlAtEvalPoints,
                    elemOrts,
                    &basis,
                    &projStruct);
              }
            }

            //check that fun values at reference points coincide with those computed using basis functions
            DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
            DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
            DynRankView basisValuesAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords, dim);


            DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords, dim);
            basis.getValues(basisValuesAtRefCoords, refPoints);
            rst::clone(basisValuesAtRefCoordsCells,basisValuesAtRefCoords);

            using namespace std;
            for (int j=0; j<basisCardinality; j++)
            {
              for (int pt=0; pt<numRefCoords; pt++)
              {
                for (int d0=0; d0<dim; d0++)
                {
                  cout << "basisValuesAtRefCoords(" << j << "," << pt << "," << d0 << "): " <<  basisValuesAtRefCoords(j,pt,d0) << endl;
                }
              }
            }
            
            // modify basis values to account for orientations
            ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
                basisValuesAtRefCoordsCells,
                elemOrts,
                &basis);
            
            // DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords, dim);
            for (int i=0; i<numElems; i++)
            {
              for (int j=0; j<basisCardinality; j++)
              {
                for (int pt=0; pt<numRefCoords; pt++)
                {
                  for (int d0=0; d0<dim; d0++)
                  {
                    cout << "basisValuesAtRefCoordsOriented(" << i <<"," << j << "," << pt << "," << d0 << "): " <<  basisValuesAtRefCoordsOriented(i,j,pt,d0) << endl;
                  }
                }
              }
            }

            // transform basis values to the reference element (pullback)
            DynRankView ConstructWithLabel(jacobianAtRefCoords, numElems, numRefCoords, dim, dim);
            DynRankView ConstructWithLabel(jacobianAtRefCoords_inv, numElems, numRefCoords, dim, dim);
            DynRankView ConstructWithLabel(jacobianAtRefCoords_det, numElems, numRefCoords);
            ct::setJacobian(jacobianAtRefCoords, refPoints, physVertexes, cellTopo);
            ct::setJacobianInv (jacobianAtRefCoords_inv, jacobianAtRefCoords);
            ct::setJacobianDet (jacobianAtRefCoords_det, jacobianAtRefCoords);
            fst::HCURLtransformVALUE(transformedBasisValuesAtRefCoordsOriented,
                jacobianAtRefCoords_inv,
                basisValuesAtRefCoordsOriented);

    //        using namespace std;
    //        for (int i=0; i<numElems; i++)
    //        {
    //          for (int pt=0; pt<numRefCoords; pt++)
    //          {
    //            for (int d0=0; d0<dim; d0++)
    //            {
    //              for (int d1=0; d1<dim; d1++)
    //              {
    //                cout << "jacobianAtRefCoords(" << i <<"," << pt << "," << d0 << "," << d1 << "): " <<  jacobianAtRefCoords(i,pt,d0,d1) << endl;
    //              }
    //            }
    //          }
    //        }
    //        for (int i=0; i<numElems; i++)
    //        {
    //          for (int pt=0; pt<numRefCoords; pt++)
    //          {
    //            for (int d0=0; d0<dim; d0++)
    //            {
    //              for (int d1=0; d1<dim; d1++)
    //              {
    //                cout << "jacobianAtRefCoords_inv(" << i <<"," << pt << "," << d0 << "," << d1 << "): " <<  jacobianAtRefCoords_inv(i,pt,d0,d1) << endl;
    //              }
    //            }
    //          }
    //        }
    //        for (int i=0; i<numElems; i++)
    //        {
    //          for (int pt=0; pt<numRefCoords; pt++)
    //          {
    //            cout << "jacobianAtRefCoords_det(" << i <<"," << pt << "): " <<  jacobianAtRefCoords_det(i,pt) << endl;
    //          }
    //        }

            DynRankView ConstructWithLabel(basisCurlsAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
            DynRankView ConstructWithLabel(transformedBasisCurlsAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
            DynRankView basisCurlsAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords, dim);

            DynRankView ConstructWithLabel(basisCurlsAtRefCoords, basisCardinality, numRefCoords, dim);
            basis.getValues(basisCurlsAtRefCoords, refPoints,OPERATOR_CURL);
            rst::clone(basisCurlsAtRefCoordsCells,basisCurlsAtRefCoords);

            // modify basis values to account for orientations
            ots::modifyBasisByOrientation(basisCurlsAtRefCoordsOriented,
                basisCurlsAtRefCoordsCells,
                elemOrts,
                &basis);
            
            fst::HCURLtransformCURL(transformedBasisCurlsAtRefCoordsOriented,
                jacobianAtRefCoords,
                jacobianAtRefCoords_det,
                basisCurlsAtRefCoordsOriented);

            DynRankView ConstructWithLabel(projectedFunAtRefCoords, numElems, numRefCoords, dim);
            DynRankView ConstructWithLabel(funCurlAtRefCoordsOriented, numElems, numRefCoords,dim);

    //        for (int i=0; i<numElems; i++)
    //        {
    //          for(ordinal_type k=0; k<basisCardinality; ++k) {
    //            cout << "basisCoeffsHCurl(" << i << "," << k << "): " << basisCoeffsHCurl(i,k) << endl;
    //          }
    //        }
    //        for (int i=0; i<numElems; i++)
    //        {
    //          for(ordinal_type j=0; j<numRefCoords; ++j)
    //          {
    //            for(ordinal_type k=0; k<basisCardinality; ++k) {
    //              for(ordinal_type d=0; d<dim; ++d) {
    //                cout << "transformedBasisValuesAtRefCoordsOriented(" << i << "," << k << "," << j << "," << d << "): " << transformedBasisValuesAtRefCoordsOriented(i,k,j,d) << endl;
    //              }
    //            }
    //          }
    //        }
            
            //compute error of projection in HCURL norm
            ValueType norm2(0);
            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &i, double &norm2Update) {
              for(ordinal_type j=0; j<numRefCoords; ++j)
                for(ordinal_type d=0; d<dim; ++d) {
                  for(ordinal_type k=0; k<basisCardinality; ++k) {
                    projectedFunAtRefCoords(i,j,d) += basisCoeffsHCurl(i,k)*transformedBasisValuesAtRefCoordsOriented(i,k,j,d);
                    funCurlAtRefCoordsOriented(i,j,d) += basisCoeffsHCurl(i,k)*transformedBasisCurlsAtRefCoordsOriented(i,k,j,d);
                  }

                  const auto funAtPRC   = funAtPhysRefCoords(i,j,d);
                  const auto jacAtRCdet = jacobianAtRefCoords_det(i,j);
                  const auto projFunAtPRC = projectedFunAtRefCoords(i,j,d);
                  const auto funCurlAtPRC = funCurlAtPhysRefCoords(i,j,d);
                  const auto funCurlAtRCO = funCurlAtRefCoordsOriented(i,j,d);
                  const auto weight = weights(j);
                  
                  cout << "i = " << i << ", j = " << j << ", d = " << d << endl;
                  cout << "funAtPRC   = " << funAtPRC << endl;
                  cout << "jacAtRCdet = " << jacAtRCdet << endl;
                  cout << "projFunAtPRC = " << projFunAtPRC << endl;
                  cout << "funCurlAtPRC = " << funCurlAtPRC << endl;
                  cout << "funCurlAtRCO = " << funCurlAtRCO << endl;
                  cout << "weight = " << weight << endl;
                  
                  const auto absJacobianDet = (jacobianAtRefCoords_det(i,j) < 0) ? -jacobianAtRefCoords_det(i,j) : jacobianAtRefCoords_det(i,j);
                  norm2Update += (funAtPhysRefCoords(i,j,d) - projectedFunAtRefCoords(i,j,d))*
                      (funAtPhysRefCoords(i,j,d) - projectedFunAtRefCoords(i,j,d))*
                      weights(j)*absJacobianDet;
                  norm2Update += (funCurlAtPhysRefCoords(i,j,d) - funCurlAtRefCoordsOriented(i,j,d))*
                      (funCurlAtPhysRefCoords(i,j,d) - funCurlAtRefCoordsOriented(i,j,d))*
                      weights(j)*absJacobianDet;
                }
            },norm2);
            ExecSpaceType().fence();

            hcurlNorm[iter] =  std::sqrt(norm2);
            auto expected_error = useL2Projection ? hcurl_errors_L2[iter] : hcurl_errors[iter];
            if (std::isnan(hcurlNorm[iter]))
            {
              errorFlag++;
//              *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
//              *outStream << "For N = " << NX << ", computed error (" << hcurlNorm[iter] << ") is nan!";
//              *outStream << std::endl;
              *outStream << "*** FAILURE (NAN): p=" << p << ", dofOrdinal " << dofOrdinal << " ***\n";
            }
            else if(std::abs(hcurlNorm[iter]-expected_error) > tol){
              errorFlag++;
//              *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
//              *outStream << "For N = " << NX << ", computed error (" << hcurlNorm[iter] << ") is different than expected one (" << expected_error << ")";
//              *outStream << std::endl;
              *outStream << "*** FAILURE: p=" << p << ", dofOrdinal " << dofOrdinal << " ***\n";
            }
            else
            {
              *outStream << "*** Success: p=" << p << ", dofOrdinal " << dofOrdinal << " ***\n";
            }
            
            if(useL2Projection)
              *outStream << "HCURL Error (L2 Projection): " << hcurlNorm[iter] <<std::endl;
            else
              *outStream << "HCURL Error (HCurl Projection): " << hcurlNorm[iter] <<std::endl;
            }
          } // basisCardinality
        } // p
      } // 
    } catch (std::exception &err) {
      std::cout << " Exception\n";
      *outStream << err.what() << "\n\n";
      errorFlag = -1000;
    }
     
    *outStream
    << "===============================================================================\n"
    << "|                                                                             |\n"
    << "|                 Test 3 (Patch Projection - HDIV)                            |\n"
    << "|                                                                             |\n"
    << "===============================================================================\n";


    try {
      // compute orientations for cells (one time computation)
      Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numElems);
      cellGeometry.orientations(elemOrts);

      for (auto useL2Projection:useL2Proj) { //

      std::vector<basisType*> basis_set;
      basis_set.push_back(new typename  CG_HBasis::HDIV_PYR(basisDegree));

      for (auto basisPtr:basis_set) {
        auto& basis = *basisPtr;
        *outStream << " " << basis.getName() << std::endl;
        ordinal_type basisCardinality = basis.getCardinality();

        //Compute physical Dof Coordinates and Reference coordinates
        DynRankView ConstructWithLabel(physRefCoords, numElems, numRefCoords, dim);
        DynRankView ConstructWithLabel(physDofCoords, numElems, basisCardinality, dim);
        {
          Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> linearBasis; //used for computing physical coordinates
          DynRankView ConstructWithLabel(linearBasisValuesAtRefCoords, numNodesPerElem, numRefCoords);
          linearBasis.getValues(linearBasisValuesAtRefCoords, refPoints);
          ExecSpaceType().fence();
          Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
          KOKKOS_LAMBDA (const int &i) {
            for(ordinal_type d=0; d<dim; ++d)
              for(ordinal_type j=0; j<numRefCoords; ++j)
                for(ordinal_type k=0; k<numNodesPerElem; ++k)
                  physRefCoords(i,j,d) += cellGeometry(i,k,d)*linearBasisValuesAtRefCoords(k,j);
          });
          ExecSpaceType().fence();
        }

        DynRankView ConstructWithLabel(funAtRefCoords, numElems, numRefCoords, dim);
        DynRankView ConstructWithLabel(funDivAtPhysRefCoords, numElems, numRefCoords);
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
        KOKKOS_LAMBDA (const int &i) {
          FunDiv fun {basisDegree};
          DivFunDiv funDiv {basisDegree};
          for(ordinal_type j=0; j<numRefCoords; ++j) {
            funDivAtPhysRefCoords(i,j) = funDiv(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2));
            for(ordinal_type k=0; k<dim; ++k)
              funAtRefCoords(i,j,k) = fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2), k);
          }
        });
        ExecSpaceType().fence();

        // compute projection-based interpolation of fun into HDIV
        DynRankView ConstructWithLabel(basisCoeffsHDiv, numElems, basisCardinality);
        {
          ordinal_type targetCubDegree(basis.getDegree()),targetDerivCubDegree(basis.getDegree()-1);

          Experimental::ProjectionStruct<DeviceType,ValueType> projStruct;
          if(useL2Projection) {
            projStruct.createL2ProjectionStruct(&basis, targetCubDegree);
          } else {
            projStruct.createHDivProjectionStruct(&basis, targetCubDegree, targetDerivCubDegree);
          }

          auto evaluationPoints = projStruct.getAllEvalPoints();
          auto evaluationDivPoints = projStruct.getAllDerivEvalPoints();
          ordinal_type numPoints = evaluationPoints.extent(0), numDivPoints = evaluationDivPoints.extent(0);

          DynRankView ConstructWithLabel(targetAtEvalPoints, numElems, numPoints, dim);
          DynRankView ConstructWithLabel(targetDivAtEvalPoints, numElems, numDivPoints);


          DynRankView ConstructWithLabel(physEvalPoints, numElems, numPoints, dim);
          DynRankView ConstructWithLabel(physEvalDivPoints, numElems, numDivPoints, dim);
          {
            DynRankView ConstructWithLabel(linearBasisValuesAtEvalPoint, numElems, numNodesPerElem);
            DynRankView ConstructWithLabel(linearBasisValuesAtEvalDivPoint, numElems, numNodesPerElem);

            Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &i) {
              auto basisValuesAtEvalPoint = Kokkos::subview(linearBasisValuesAtEvalPoint,i,Kokkos::ALL());
              for(ordinal_type j=0; j<numPoints; ++j){
                auto evalPoint = Kokkos::subview(evaluationPoints,j,Kokkos::ALL());
                Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalPoint, evalPoint);
                for(ordinal_type k=0; k<numNodesPerElem; ++k)
                  for(ordinal_type d=0; d<dim; ++d)
                    physEvalPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalPoint(k);
              }

              auto basisValuesAtEvalDivPoint = Kokkos::subview(linearBasisValuesAtEvalDivPoint,i,Kokkos::ALL());
              for(ordinal_type j=0; j<numDivPoints; ++j) {
                auto evalGradPoint = Kokkos::subview(evaluationDivPoints,j,Kokkos::ALL());
                Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalDivPoint, evalGradPoint);
                for(ordinal_type k=0; k<numNodesPerElem; ++k)
                  for(ordinal_type d=0; d<dim; ++d)
                    physEvalDivPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalDivPoint(k);
              }
            });
            ExecSpaceType().fence();
          }

          //transform the target function and its derivative to the reference element (inverse of pullback operator)
          DynRankView ConstructWithLabel(jacobian, numElems, numPoints, dim, dim);
          DynRankView ConstructWithLabel(jacobian_det, numElems, numPoints);
          DynRankView ConstructWithLabel(jacobian_inv, numElems, numPoints, dim, dim);
          ct::setJacobian(jacobian, evaluationPoints, physVertexes, cellTopo);
          ct::setJacobianDet (jacobian_det, jacobian);
          ct::setJacobianInv (jacobian_inv, jacobian);

          DynRankView ConstructWithLabel(jacobianDiv_det, numElems, numDivPoints);
          if(numDivPoints>0){
            DynRankView ConstructWithLabel(jacobianDiv, numElems, numDivPoints, dim, dim);
            ct::setJacobian(jacobianDiv, evaluationDivPoints, physVertexes, cellTopo);
            ct::setJacobianDet (jacobianDiv_det, jacobianDiv);
          }

          Kokkos::deep_copy(targetDivAtEvalPoints,0.);
          Kokkos::deep_copy(targetAtEvalPoints,0.);
          Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
          KOKKOS_LAMBDA (const int &ic) {
            FunDiv fun {basisDegree};
            DivFunDiv divFun {basisDegree};
            for(int i=0;i<numPoints;i++) {
              for(int j=0;j<dim;j++)
                for(int d=0;d<dim;d++)
                  targetAtEvalPoints(ic,i,j) += jacobian_det(ic,i)*jacobian_inv(ic,i,j,d)*fun(physEvalPoints(ic,i,0), physEvalPoints(ic,i,1), physEvalPoints(ic,i,2),d);
            }
            for(int i=0;i<numDivPoints;i++) {
              targetDivAtEvalPoints(ic,i) += jacobianDiv_det(ic,i)*divFun(physEvalDivPoints(ic,i,0), physEvalDivPoints(ic,i,1), physEvalDivPoints(ic,i,2));//funHGradCoeffs(k)
            }
          });
          ExecSpaceType().fence();

          if(useL2Projection) {
            pts::getL2BasisCoeffs(basisCoeffsHDiv,
                targetAtEvalPoints,
                elemOrts,
                &basis,
                &projStruct);
          } else {
            pts::getHDivBasisCoeffs(basisCoeffsHDiv,
                targetAtEvalPoints,
                targetDivAtEvalPoints,
                elemOrts,
                &basis,
                &projStruct);
          }
        }

        //check that fun values at reference points coincide with those computed using basis functions
        DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
        DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords, dim);
        DynRankView basisValuesAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords, dim);

        DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords, dim);
        basis.getValues(basisValuesAtRefCoords, refPoints);
        rst::clone(basisValuesAtRefCoordsCells,basisValuesAtRefCoords);

        // modify basis values to account for orientations
        ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
            basisValuesAtRefCoordsCells,
            elemOrts,
            &basis);

        // transform basis values to the reference element (pullback)
        DynRankView ConstructWithLabel(jacobianAtRefCoords, numElems, numRefCoords, dim, dim);
        DynRankView ConstructWithLabel(jacobianAtRefCoords_det, numElems, numRefCoords);
        ct::setJacobian(jacobianAtRefCoords, refPoints, physVertexes, cellTopo);
        ct::setJacobianDet (jacobianAtRefCoords_det, jacobianAtRefCoords);
        fst::HDIVtransformVALUE(transformedBasisValuesAtRefCoordsOriented,
            jacobianAtRefCoords,
            jacobianAtRefCoords_det,
            basisValuesAtRefCoordsOriented);

        DynRankView ConstructWithLabel(basisDivsAtRefCoordsOriented, numElems, basisCardinality, numRefCoords);
        DynRankView ConstructWithLabel(transformedBasisDivsAtRefCoordsOriented, numElems, basisCardinality, numRefCoords);
        DynRankView basisDivsAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords);

        DynRankView ConstructWithLabel(basisDivsAtRefCoords, basisCardinality, numRefCoords);
        basis.getValues(basisDivsAtRefCoords, refPoints,OPERATOR_DIV);
        rst::clone(basisDivsAtRefCoordsCells,basisDivsAtRefCoords);

        // modify basis values to account for orientations
        ots::modifyBasisByOrientation(basisDivsAtRefCoordsOriented,
            basisDivsAtRefCoordsCells,
            elemOrts,
            &basis);


        fst::HDIVtransformDIV(transformedBasisDivsAtRefCoordsOriented,
            jacobianAtRefCoords_det,
            basisDivsAtRefCoordsOriented);


        DynRankView ConstructWithLabel(projectedFunAtRefCoords, numElems, numRefCoords, dim);
        DynRankView ConstructWithLabel(funDivAtRefCoordsOriented, numElems, numRefCoords);

        //compute error of projection in HDIV norm
        ValueType norm2(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
        KOKKOS_LAMBDA (const int &i, double &norm2Update) {
          for(ordinal_type j=0; j<numRefCoords; ++j) {
            for(ordinal_type k=0; k<basisCardinality; ++k) {
              for(ordinal_type d=0; d<dim; ++d)
                projectedFunAtRefCoords(i,j,d) += basisCoeffsHDiv(i,k)*transformedBasisValuesAtRefCoordsOriented(i,k,j,d);
              funDivAtRefCoordsOriented(i,j) += basisCoeffsHDiv(i,k)*transformedBasisDivsAtRefCoordsOriented(i,k,j);
            }

            const auto absJacobianDet = (jacobianAtRefCoords_det(i,j) < 0) ? -jacobianAtRefCoords_det(i,j) : jacobianAtRefCoords_det(i,j);
            for(ordinal_type d=0; d<dim; ++d) {
              norm2Update += (funAtRefCoords(i,j,d) - projectedFunAtRefCoords(i,j,d))*
                  (funAtRefCoords(i,j,d) - projectedFunAtRefCoords(i,j,d))*
                  weights(j)*absJacobianDet;
            }
            norm2Update += (funDivAtPhysRefCoords(i,j) - funDivAtRefCoordsOriented(i,j))*
                (funDivAtPhysRefCoords(i,j) - funDivAtRefCoordsOriented(i,j))*
                weights(j)*absJacobianDet;
          }
        },norm2);
        ExecSpaceType().fence();
        hdivNorm[iter] = std::sqrt(norm2);
        auto expected_error = useL2Projection ? hdiv_errors_L2[iter] : hdiv_errors[iter];
        if (std::isnan(hdivNorm[iter]))
        {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "For N = " << NX << ", computed error (" << hdivNorm[iter] << ") is nan!";
          *outStream << std::endl;
        }
        else if(std::abs(hdivNorm[iter]-expected_error) > tol){
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "For N = " << NX << ", computed error (" << hdivNorm[iter] << ") is different than expected one (" << expected_error << ")";
          *outStream << std::endl;
        }
        delete basisPtr;
      }
      if(useL2Projection)
        *outStream << "HDIV Error (L2 Projection): " << hdivNorm[iter] <<std::endl;
      else
        *outStream << "HDIV Error (HDiv Projection): " << hdivNorm[iter] <<std::endl;
      }
    } catch (std::exception &err) {
      std::cout << " Exception\n";
      *outStream << err.what() << "\n\n";
      errorFlag = -1000;
    }



    *outStream
    << "===============================================================================\n"
    << "|                                                                             |\n"
    << "|                 Test 4 (Patch Projection - HVOL)                            |\n"
    << "|                                                                             |\n"
    << "===============================================================================\n";


    try {
      // compute orientations for cells (one-time computation)
      Kokkos::DynRankView<Orientation,DeviceType> elemOrts("elemOrts", numElems);
      cellGeometry.orientations(elemOrts);

      for (auto useL2Projection:useL2Proj) { //
      std::vector<basisType*> basis_set;
      basis_set.push_back(new typename  CG_HBasis::HVOL_PYR(basisDegree));

      for (auto basisPtr:basis_set) {
        auto& basis = *basisPtr;
        *outStream << " " << basis.getName() << std::endl;
        ordinal_type basisCardinality = basis.getCardinality();

        //Compute physical Dof Coordinates and Reference coordinates
        DynRankView ConstructWithLabel(physRefCoords, numElems, numRefCoords, dim);
        {
          Basis_HGRAD_PYR_C1_FEM<DeviceType,ValueType,ValueType> linearBasis; //used for computing physical coordinates
          DynRankView ConstructWithLabel(linearBasisValuesAtRefCoords, numNodesPerElem, numRefCoords);
          linearBasis.getValues(linearBasisValuesAtRefCoords, refPoints);
          ExecSpaceType().fence();
          Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
          KOKKOS_LAMBDA (const int &i) {
            for(ordinal_type d=0; d<dim; ++d)
              for(ordinal_type j=0; j<numRefCoords; ++j)
                for(ordinal_type k=0; k<numNodesPerElem; ++k)
                  physRefCoords(i,j,d) += cellGeometry(i,k,d)*linearBasisValuesAtRefCoords(k,j);
          });
          ExecSpaceType().fence();
        }

        //check function reproducibility
        DynRankView ConstructWithLabel(funAtRefCoords, numElems, numRefCoords);
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
        KOKKOS_LAMBDA (const int &i) {
          Fun fun {basisDegree};
          for(ordinal_type j=0; j<numRefCoords; ++j)
            funAtRefCoords(i,j) = fun(physRefCoords(i,j,0), physRefCoords(i,j,1), physRefCoords(i,j,2));
        });
        ExecSpaceType().fence();

        // compute projection-based interpolation of fun into HVOL
        DynRankView ConstructWithLabel(basisCoeffsHVol, numElems, basisCardinality);
        {
          ordinal_type targetCubDegree(basis.getDegree());

          Experimental::ProjectionStruct<DeviceType,ValueType> projStruct;
          if(useL2Projection) {
            projStruct.createL2ProjectionStruct(&basis, targetCubDegree);
          } else {
            projStruct.createHVolProjectionStruct(&basis, targetCubDegree);
          }

          auto evaluationPoints = projStruct.getAllEvalPoints();
          ordinal_type numPoints = evaluationPoints.extent(0);

          DynRankView ConstructWithLabel(targetAtEvalPoints, numElems, numPoints);


          DynRankView ConstructWithLabel(physEvalPoints, numElems, numPoints, dim);
          {
            DynRankView ConstructWithLabel(linearBasisValuesAtEvalPoint, numElems, numNodesPerElem);

            Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
            KOKKOS_LAMBDA (const int &i) {
              auto basisValuesAtEvalPoint = Kokkos::subview(linearBasisValuesAtEvalPoint,i,Kokkos::ALL());
              for(ordinal_type j=0; j<numPoints; ++j){
                auto evalPoint = Kokkos::subview(evaluationPoints,j,Kokkos::ALL());
                Impl::Basis_HGRAD_PYR_C1_FEM::template Serial<OPERATOR_VALUE>::getValues(basisValuesAtEvalPoint, evalPoint);
                for(ordinal_type k=0; k<numNodesPerElem; ++k)
                  for(ordinal_type d=0; d<dim; ++d)
                    physEvalPoints(i,j,d) += cellGeometry(i,k,d)*basisValuesAtEvalPoint(k);
              }
            });
            ExecSpaceType().fence();
          }

          //transform the target function to the reference element (inverse of pullback operator)
          DynRankView ConstructWithLabel(jacobian, numElems, numPoints, dim, dim);
          DynRankView ConstructWithLabel(jacobian_det, numElems, numPoints);
          ct::setJacobian(jacobian, evaluationPoints, physVertexes, cellTopo);
          ct::setJacobianDet (jacobian_det, jacobian);

          Kokkos::deep_copy(targetAtEvalPoints,0.);
          Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
          KOKKOS_LAMBDA (const int &ic) {
            Fun fun {basisDegree};
            for(int i=0;i<numPoints;i++)
              targetAtEvalPoints(ic,i) += jacobian_det(ic,i)*fun(physEvalPoints(ic,i,0), physEvalPoints(ic,i,1), physEvalPoints(ic,i,2));
          });
          ExecSpaceType().fence();
          if(useL2Projection) {
            pts::getL2BasisCoeffs(basisCoeffsHVol,
                targetAtEvalPoints,
                elemOrts,
                &basis,
                &projStruct);
          } else {
            pts::getHVolBasisCoeffs(basisCoeffsHVol,
                targetAtEvalPoints,
                elemOrts,
                &basis,
                &projStruct);
          }
        }


        //check that fun values at reference points coincide with those computed using basis functions
        DynRankView ConstructWithLabel(basisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords);
        DynRankView ConstructWithLabel(transformedBasisValuesAtRefCoordsOriented, numElems, basisCardinality, numRefCoords);
        DynRankView basisValuesAtRefCoordsCells("inValues", numElems, basisCardinality, numRefCoords);

        DynRankView ConstructWithLabel(basisValuesAtRefCoords, basisCardinality, numRefCoords);
        basis.getValues(basisValuesAtRefCoords, refPoints);
        rst::clone(basisValuesAtRefCoordsCells,basisValuesAtRefCoords);

        // modify basis values to account for orientations
        ots::modifyBasisByOrientation(basisValuesAtRefCoordsOriented,
            basisValuesAtRefCoordsCells,
            elemOrts,
            &basis);

        // transform basis values to the reference element (pullback)
        DynRankView ConstructWithLabel(jacobianAtRefCoords, numElems, numRefCoords, dim, dim);
        DynRankView ConstructWithLabel(jacobianAtRefCoords_det, numElems, numRefCoords);
        ct::setJacobian(jacobianAtRefCoords, refPoints, physVertexes, cellTopo);
        ct::setJacobianDet (jacobianAtRefCoords_det, jacobianAtRefCoords);
        fst::HVOLtransformVALUE(transformedBasisValuesAtRefCoordsOriented,
            jacobianAtRefCoords_det,
            basisValuesAtRefCoordsOriented);

        DynRankView ConstructWithLabel(projectedFunAtRefCoords, numElems, numRefCoords);

        //compute error of projection in L2 norm
        ValueType norm2(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpaceType>(0,numElems),
        KOKKOS_LAMBDA (const int &i, double &norm2Update) {
          for(ordinal_type j=0; j<numRefCoords; ++j) {
            for(ordinal_type k=0; k<basisCardinality; ++k)
              projectedFunAtRefCoords(i,j) += basisCoeffsHVol(i,k)*transformedBasisValuesAtRefCoordsOriented(i,k,j);
            const auto absJacobianDet = (jacobianAtRefCoords_det(i,j) < 0) ? -jacobianAtRefCoords_det(i,j) : jacobianAtRefCoords_det(i,j);
            norm2Update += (funAtRefCoords(i,j) - projectedFunAtRefCoords(i,j))*
                (funAtRefCoords(i,j) - projectedFunAtRefCoords(i,j))*
                weights(j)*absJacobianDet;
          }
        },norm2);
        ExecSpaceType().fence();

        hvolNorm[iter] =  std::sqrt(norm2);
        auto expected_error = useL2Projection ? hvol_errors_L2[iter] : hvol_errors[iter];
        if (std::isnan(hvolNorm[iter]))
        {
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "For N = " << NX << ", computed error (" << hvolNorm[iter] << ") is nan!";
          *outStream << std::endl;
        }
        else if(std::abs(hvolNorm[iter]-expected_error) > tol){
          errorFlag++;
          *outStream << std::setw(70) << "^^^^----FAILURE!" << "\n";
          *outStream << "For N = " << NX << ", computed error (" << hvolNorm[iter] << ") is different than expected one (" << expected_error << ")";
          *outStream << std::endl;
        }
        delete basisPtr;
      }
      if(useL2Projection)
        *outStream << "HVOL Error (L2 Projection): " << hvolNorm[iter] <<std::endl;
      else
        *outStream << "HVOL Error (HVol Projection): " << hvolNorm[iter] <<std::endl;
      }
    } catch (std::exception &err) {
      std::cout << " Exception\n";
      *outStream << err.what() << "\n\n";
      errorFlag = -1000;
    }
  }
/*
  *outStream << "\nHGRAD ERROR:";
  for(int iter = 0; iter<numRefinements; iter++)
    *outStream << " " << hgradNorm[iter];
  *outStream << "\nHCURL ERROR:";
  for(int iter = 0; iter<numRefinements; iter++)
    *outStream << " " << hcurlNorm[iter];
  *outStream << "\nHDIV ERROR:";
  for(int iter = 0; iter<numRefinements; iter++)
    *outStream << " " << hdivNorm[iter];
  *outStream << "\nHVOL ERROR:";
  for(int iter = 0; iter<numRefinements; iter++)
    *outStream << " " << hvolNorm[iter];
  *outStream << std::endl;
*/

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED = " << errorFlag << "\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  // reset format state of std::cout
  std::cout.copyfmt(oldFormatState);
  return errorFlag;
}
}
}

