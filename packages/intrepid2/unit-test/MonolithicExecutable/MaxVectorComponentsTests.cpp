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
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov),
//                    Mauro Perego  (mperego@sandia.gov), or
//                    Nate Roberts  (nvrober@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   CellGeometryTests.cpp
    \brief  Tests against CellGeometry
    \author Created by N.V. Roberts.
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_CellGeometry.hpp"
#include "Intrepid2_CellGeometryTestUtils.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_ProjectedGeometry.hpp"
#include "Intrepid2_ProjectedGeometryExamples.hpp"
#include "Intrepid2_ScalarView.hpp"
#include "Intrepid2_Types.hpp"
#include "Intrepid2_TestUtils.hpp"

#include <Kokkos_Core.hpp>

using namespace Intrepid2;

namespace
{
  template<class HGRAD_LINE>
  class Basis_SomeDirectSumBasis
  : public Basis_DirectSumBasis <typename HGRAD_LINE::BasisBase>
  {
    using Family1 = HGRAD_LINE;
    using Family2 = HGRAD_LINE;
    using DirectSumBasis = Basis_DirectSumBasis <typename HGRAD_LINE::BasisBase>;
    
    using BasisBase = typename HGRAD_LINE::BasisBase;
    
    using DeviceType = typename HGRAD_LINE::DeviceType;

  protected:
    std::string name_;
    ordinal_type order_x_;

  public:
    using ExecutionSpace  = typename HGRAD_LINE::ExecutionSpace;
    using OutputValueType = typename HGRAD_LINE::OutputValueType;
    using PointValueType  = typename HGRAD_LINE::PointValueType;
    
    /** \brief  Constructor.
        \param [in] polyOrder_x - the polynomial order in the x dimension.
        \param [in] polyOrder_y - the polynomial order in the y dimension.
        \param [in] pointType   - type of lattice used for creating the DoF coordinates.
     */
    Basis_SomeDirectSumBasis(int polyOrder_x)
    :
    DirectSumBasis(Teuchos::rcp( new Family1(polyOrder_x) ),
                   Teuchos::rcp( new Family2(polyOrder_x) ))
    {
      this->functionSpace_ = FUNCTION_SPACE_HGRAD;

      std::ostringstream basisName;
      basisName << "Some DirectSumBasis (" << this->DirectSumBasis::getName() << ")";
      name_ = basisName.str();

      order_x_ = polyOrder_x;
    }
    
    BasisValues<OutputValueType,DeviceType> allocateBasisValues2( TensorPoints<PointValueType,DeviceType> points, const EOperator operatorType = OPERATOR_VALUE) const
    {
      BasisValues<OutputValueType,DeviceType> basisValues;
      
      BasisValues<OutputValueType,DeviceType> basisValues1 = this->basis1_->allocateBasisValues(points, operatorType);
      BasisValues<OutputValueType,DeviceType> basisValues2 = this->basis2_->allocateBasisValues(points, operatorType);
    
      return basisValues;
    }
    
    BasisValues<OutputValueType,DeviceType> allocateBasisValues( TensorPoints<PointValueType,DeviceType> points, const EOperator operatorType = OPERATOR_VALUE) const override
    {
      BasisValues<OutputValueType,DeviceType> basisValues1 = this->basis1_->allocateBasisValues(points, operatorType);
      BasisValues<OutputValueType,DeviceType> basisValues2 = this->basis2_->allocateBasisValues(points, operatorType);

      BasisValues<OutputValueType,DeviceType> basisValues;
      
      const int numScalarFamilies1 = basisValues1.numTensorDataFamilies();
      if (numScalarFamilies1 > 0)
      {
        // then both basis1 and basis2 should be scalar-valued; check that for basis2:
        const int numScalarFamilies2 = basisValues2.numTensorDataFamilies();
        INTREPID2_TEST_FOR_EXCEPTION(basisValues2.numTensorDataFamilies() <=0, std::invalid_argument, "When basis1 has scalar value, basis2 must also");
        std::vector< TensorData<OutputValueType,DeviceType> > scalarFamilies(numScalarFamilies1 + numScalarFamilies2);
        for (int i=0; i<numScalarFamilies1; i++)
        {
          scalarFamilies[i] = basisValues1.tensorData(i);
        }
        for (int i=0; i<numScalarFamilies2; i++)
        {
          scalarFamilies[i+numScalarFamilies1] = basisValues2.tensorData(i);
        }
        basisValues = BasisValues<OutputValueType,DeviceType>(scalarFamilies);
      }
      else
      {
        // then both basis1 and basis2 should be vector-valued; check that:
        INTREPID2_TEST_FOR_EXCEPTION(!basisValues1.vectorData().isValid(), std::invalid_argument, "When basis1 does not have tensorData() defined, it must have a valid vectorData()");
        INTREPID2_TEST_FOR_EXCEPTION(basisValues2.numTensorDataFamilies() > 0, std::invalid_argument, "When basis1 has vector value, basis2 must also");

        const auto & vectorData1 = basisValues1.vectorData();
        const auto & vectorData2 = basisValues2.vectorData();

        const int numFamilies1  = vectorData1.numFamilies();
        const int numComponents = vectorData1.numComponents();
        INTREPID2_TEST_FOR_EXCEPTION(numComponents != vectorData2.numComponents(), std::invalid_argument, "basis1 and basis2 must agree on the number of components in each vector");
        const int numFamilies2 = vectorData2.numFamilies();

        const int numFamilies = numFamilies1 + numFamilies2;
        std::vector< std::vector<TensorData<OutputValueType,DeviceType> > > vectorComponents(numFamilies, std::vector<TensorData<OutputValueType,DeviceType> >(numComponents));

        for (int i=0; i<numFamilies1; i++)
        {
          for (int j=0; j<numComponents; j++)
          {
            vectorComponents[i][j] = vectorData1.getComponent(i,j);
          }
        }
        for (int i=0; i<numFamilies2; i++)
        {
          for (int j=0; j<numComponents; j++)
          {
            vectorComponents[i+numFamilies1][j] = vectorData2.getComponent(i,j);
          }
        }
        VectorData<OutputValueType,DeviceType> vectorData(vectorComponents);
        basisValues = BasisValues<OutputValueType,DeviceType>(vectorData);
      }
      return basisValues;
    }
    
    /** \brief True if orientation is required
    */
    virtual bool requireOrientation() const override
    {
      return (this->getDofCount(1,0) > 0); //if it has edge DOFs, than it needs orientations
    }

    /** \brief  Returns basis name

     \return the name of the basis
     */
    virtual
    const char*
    getName() const override {
      return name_.c_str();
    }

    /** \brief Creates and returns a Basis object whose DeviceType template argument is Kokkos::HostSpace::device_type, but is otherwise identical to this.
     
        \return Pointer to the new Basis object.
     */
    virtual HostBasisPtr<OutputValueType, PointValueType>
    getHostBasis() const override {
      using HostBasis  = Basis_SomeDirectSumBasis<typename HGRAD_LINE::HostBasis>;
      
      auto hostBasis = Teuchos::rcp(new HostBasis(order_x_));
      
      return hostBasis;
    }
  };

  TEUCHOS_UNIT_TEST( MaxVectorComponents, SegFaultCompilerIssueMaybe )
  {
    using DeviceType = Intrepid2::DefaultTestDeviceType;
    
    using LineBasis = IntegratedLegendreBasis_HGRAD_LINE<DeviceType,double,double,true>;
    using Basis = Basis_SomeDirectSumBasis<LineBasis>;
    
    std::cout << "\n\n";
    std::cout << "sizeof(BasisValues<double,DeviceType>) = " << sizeof(BasisValues<double,DeviceType>) << std::endl;
    
    const int polyOrder = 1;
    Basis basis(polyOrder);
    TensorPoints<double,DeviceType> tensorPoints;
    auto basisValues = basis.allocateBasisValues2(tensorPoints, OPERATOR_VALUE);
    
    if (success)
    {
      std::cout << "\n\n******************** TO REPRODUCE ISSUE ************************\n";
      std::cout << "If you get here, you haven't hit the issue.  Last confirmed using\n";
      std::cout << "Apple clang version 13.0.0 (clang-1300.0.29.3).  To reproduce, \n";
      std::cout << "set MaxVectorComponents to 66.\n";
      std::cout << "****************************************************************\n\n";
    }
  }
} // namespace
