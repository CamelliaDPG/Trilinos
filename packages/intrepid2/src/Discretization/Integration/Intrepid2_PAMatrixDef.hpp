// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_PAMatrixDef.hpp
    \brief  Header file for the Intrepid2::PAMatrix implementations; provides support for matrix partial assembly.
    \author Created by Nathan V. Roberts.
*/

#ifndef __INTREPID2_PAMATRIX_DEF_HPP__
#define __INTREPID2_PAMATRIX_DEF_HPP__

#include "Intrepid2_PAMatrix.hpp"

#include "Intrepid2_DataDimensionInfo.hpp"
#include "Intrepid2_OrientationTools.hpp"

namespace Intrepid2 {

namespace Impl
{
  template<typename DeviceType,class Scalar>
  class GemmSequenceFunctor
  {
  public:
    using ExecutionSpace = typename DeviceType::execution_space;
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TeamMember = typename TeamPolicy::member_type;
    
    int numCells_;
    int numPoints_;
    int D1_ = -1;    // first dimension of pointwise data, if it is vector- or matrix-valued
    int D2_ = -1;    // second dimension of pointwise data, if it is matrix-valued
    int numVectors_; // N, the number of vectors in the multi-vector input/output
    int inputSize_;  // total number of  input data entries per cell
    int outputSize_; // total number of output data entries per cell
    int maxIntermediateSize_; // the highest entry count we'll need per cell as we apply operators
    int fad_size_output_;
    
    using View2D = Kokkos::View<  Scalar**,DeviceType>;
    using View3D = Kokkos::View< Scalar***,DeviceType>;
    using View4D = Kokkos::View<Scalar****,DeviceType>;
    
    Kokkos::Array<int,8> rightOpRowDims_; // point dimension
    Kokkos::Array<int,8> rightOpColDims_; // field dimension
    
    Kokkos::Array<int,8>  leftOpRowDims_; // field dimension
    Kokkos::Array<int,8>  leftOpColDims_; // point dimension
    
    Kokkos::Array< View2D, 8> refSpaceOpsRight_; // 2D views, shape (Pj,F2j) (ith component); if operator is vector-valued or tensor-valued, the Pi dimension will be packed (so that it will have more entries than the nominal point count)
    Kokkos::Array< View2D, 8> refSpaceOpsLeft_;  // 2D views, shape (F1i,Pi)
    
    int numOpsRight_;
    int numOpsLeft_;
    
    int pointDataRank_          = -1; // 0 for scalar data, 1 for vector, 2 for matrix
    int pointExpansionFactor_   =  1; // > 1 for vector/matrix-valued point data when right vector evaluation produces a scalar
    int pointContractionFactor_ =  1; // > 1 for vector/matrix-valued point data when right vector evaluation produces a vector
    
    View3D  inputView_; // shape (C,F2,N), where F2 is the row dimension of the full operator, and N is the number of vectors
    View3D outputView_; // shape (C,F1,N), where F1 is the column dimension of the full operator
    
    //! bottleneck constructor.  Only one of scalarPointData, vectorPointData, matrixPointData may be non-empty.
    GemmSequenceFunctor(View3D outputView, View3D inputView,
                        std::vector<View2D> refSpaceOpsRight, std::vector<View2D> refSpaceOpsLeft,
                        View2D scalarPointData, View3D vectorPointData, View4D matrixPointData)
    :
    inputView_(inputView),
    outputView_(outputView)
    {
      const bool allocateFadStorage = !(std::is_standard_layout<Scalar>::value && std::is_trivial<Scalar>::value);
      if (allocateFadStorage)
      {
        fad_size_output_ = dimension_scalar(inputView_);
      }
      numOpsRight_ = int(refSpaceOpsRight.size());
      INTREPID2_TEST_FOR_EXCEPTION(numOpsRight_ > refSpaceOpsRight_.size(), std::invalid_argument, "Too many right ops");
      
      numVectors_ = inputView_.extent_int(2);
      INTREPID2_TEST_FOR_EXCEPTION(numVectors_ != outputView_.extent_int(2), std::invalid_argument, "inputView and outputView must agree on the number of vectors");
      
      inputSize_  =  inputView_.extent_int(1) * numVectors_;   // F2 * N
      outputSize_ = outputView_.extent_int(1) * numVectors_;   // F1 * N
      int maxSize = max(inputSize_,outputSize_);
      int currentSize = inputSize_;
      for (int rj=0; rj<numOpsRight_; rj++)
      {
        const auto & rightOp = refSpaceOpsRight[rj];
        refSpaceOpsRight_[rj] = rightOp;
        // right ops convert from F2j dims to Pj dims
        rightOpRowDims_[rj] = rightOp.extent_int(0); // Pj
        rightOpColDims_[rj] = rightOp.extent_int(1); // F2j
        
        currentSize = currentSize / rightOpColDims_[rj] * rightOpRowDims_[rj];
        maxSize = max(currentSize, maxSize);
      }
      if (scalarPointData.size() > 0)
      {
        pointDataRank_ = 0;
        numPoints_ = scalarPointData.extent_int(1); // (C,P)
      }
      else if (vectorPointData.size() > 0)
      {
        pointDataRank_ = 1;
        numPoints_ = vectorPointData.extent_int(1); // (C,P,D)
        D1_        = vectorPointData.extent_int(2); // (C,P,D)
        
        if (currentSize == numVectors_ * numPoints_)
        {
          pointExpansionFactor_ = D1_;
        }
        else if (currentSize == numVectors_ * numPoints_ * D1_)
        {
          pointContractionFactor_ = D1_;
        }
        else
        {
          INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "incompatible size sequence");
        }
      }
      else if (matrixPointData.size() > 0)
      {
        pointDataRank_ = 2;
        numPoints_ = matrixPointData.extent_int(1); // (C,P,D,D)
        D1_        = matrixPointData.extent_int(2); // (C,P,D,D)
        D2_        = matrixPointData.extent_int(3); // (C,P,D,D)
        
        if (currentSize == numVectors_ * numPoints_)
        {
          pointExpansionFactor_ = D1_ * D2_;
        }
        else if (currentSize == numVectors_ * numPoints_ * D2_)
        {
          // pointwise mat-vec: contract by D2, expand by D1
          pointContractionFactor_ = D2_;
          pointExpansionFactor_   = D1_;
        }
        else
        {
          INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "incompatible size sequence");
        }
      }
      else
      {
        INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "must specify scalar, vector, or matrix-valued point data");
      }
      currentSize *= pointExpansionFactor_;
      currentSize /= pointContractionFactor_;
      maxSize = max(currentSize, maxSize);
      
      numOpsLeft_  = int(refSpaceOpsLeft.size());
      INTREPID2_TEST_FOR_EXCEPTION(numOpsLeft_ > refSpaceOpsLeft_.size(), std::invalid_argument, "Too many left ops");
      for (int ri=0; ri<numOpsLeft_; ri++)
      {
        const auto & leftOp = refSpaceOpsLeft[ri];
        refSpaceOpsLeft_[ri] = leftOp;
        // left ops convert from Pi dims to F2i dims
        leftOpRowDims_[ri] = leftOp.extent_int(0); // F1i
        leftOpColDims_[ri] = leftOp.extent_int(1); // Pi
        
        currentSize = currentSize / leftOpColDims_[ri] * leftOpRowDims_[ri];
        maxSize = max(currentSize, maxSize);
      }
      INTREPID2_TEST_FOR_EXCEPTION(currentSize != outputSize_, std::invalid_argument, "Incompatible dimensions");
    }
    
    GemmSequenceFunctor(std::vector<View2D> refSpaceOpsRight, View2D scalarPointData, std::vector<View2D> refSpaceOpsLeft)
    :
    GemmSequenceFunctor(refSpaceOpsRight, refSpaceOpsLeft, scalarPointData, View3D(), View4D())
    {}
    
    GemmSequenceFunctor(std::vector<View2D> refSpaceOpsRight, View3D vectorPointData, std::vector<View2D> refSpaceOpsLeft)
    :
    GemmSequenceFunctor(refSpaceOpsRight, refSpaceOpsLeft, View2D(), vectorPointData, View4D())
    {}
    
    GemmSequenceFunctor(std::vector<View2D> refSpaceOpsRight, View4D matrixPointData, std::vector<View2D> refSpaceOpsLeft)
    :
    GemmSequenceFunctor(refSpaceOpsRight, refSpaceOpsLeft, View2D(), View3D(), matrixPointData)
    {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()( const TeamMember & teamMember ) const
    {
      const int cellOrdinal  = teamMember.league_rank();
      const int threadNumber = teamMember.team_rank();
      const int numThreads   = teamMember.team_size(); // num threads
      
      using ScratchView = Kokkos::View<Scalar*, DeviceType, Kokkos::MemoryUnmanaged>;
      
      // we alternate between using these two workspaces as the destination for the gemms
      ScratchView workspace1;
      ScratchView workspace2;
      
      if (fad_size_output_ > 0) 
      {
        workspace1 = ScratchView(teamMember.team_shmem(), maxIntermediateSize_, fad_size_output_);
        workspace2 = ScratchView(teamMember.team_shmem(), maxIntermediateSize_, fad_size_output_);
      }
      else 
      {
        workspace1 = ScratchView(teamMember.team_shmem(), maxIntermediateSize_);
        workspace2 = ScratchView(teamMember.team_shmem(), maxIntermediateSize_);
      }
      
      // TODO: sequence of gemms.  We will need to either move memory to allow the gemms to take place on the whole structure,
      //       or to specify gemms on contiguous chunks; due to the slicing of the tensor, in general the input data does not
      //       have structure of a monolithic matrix, but it does have matrix blocks whose products can be placed into a result
      //       in a memory-contiguous fashion.  Splitting into smaller gemms *might* allow expression of more parallelism, but
      //       I doubt we can beat vendor-provided implementations.
      
//      Scalar result;
//      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,0,maxFields), [&] (const int& fieldOrdinal, Scalar &contractionThusFar)
//      {
//        
//      }, result);
      
      // synchronize threads
      teamMember.team_barrier();
    }
    
  };

  //! Given (C,P[,D,D]) transform and (C,P) pointwise weights, construct a suitable container for storing the pointwise weighted transform.
  template<typename DeviceType,class Scalar>
  Data<Scalar,DeviceType> allocateComposedWeightedTransform(const Data<Scalar,DeviceType> &composedTransform,
                                                            const TensorData<Scalar,DeviceType> &pointWeights)
  {
    auto cellDimInfo = composedTransform.getDimensionInfo(0); // cell dimension
    int numTensorComponents = pointWeights.numTensorComponents();
    const int & numLogicalCells = cellDimInfo.logicalExtent;
    for (int r=0; r<numTensorComponents; r++)
    {
      auto cellDimInfo_r = pointWeights.getTensorComponent(r).getDimensionInfo(0);
      cellDimInfo = combinedDimensionInfo(cellDimInfo, cellDimInfo_r);
    }
    
    int numPoints = composedTransform.extent_int(1);
    DimensionInfo pointDimInfo {numPoints,GENERAL,numPoints,numPoints,-1};
    
    if (composedTransform.rank() == 2)
    {
      return Data<Scalar,DeviceType>({cellDimInfo,pointDimInfo});
    }
    else if (composedTransform.rank() == 3)
    {
      auto D1DimInfo = composedTransform.getDimensionInfo(2);
      return Data<Scalar,DeviceType>({cellDimInfo,pointDimInfo,D1DimInfo});
    }
    else if (composedTransform.rank() == 4)
    {
      auto D1DimInfo = composedTransform.getDimensionInfo(2);
      auto D2DimInfo = composedTransform.getDimensionInfo(3);
      return Data<Scalar,DeviceType>({cellDimInfo,pointDimInfo,D1DimInfo,D2DimInfo});
    }
    else
    {
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported rank for composedTransform");
    }
  }
} // namespace Impl

template<typename DeviceType,class Scalar>
PAMatrix<DeviceType,Scalar>::PAMatrix(const TransformedBasisValues<Scalar,DeviceType> basisValuesLeft,
                                      const TensorData<Scalar,DeviceType> cellMeasures,
                                      const TransformedBasisValues<Scalar,DeviceType> basisValuesRight,
                                      const ScalarView<Orientation,DeviceType> orientations)
:
_cellMeasures(cellMeasures),
_basisValuesLeft(basisValuesLeft),
_basisValuesRight(basisValuesRight),
_orientations(orientations)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  const bool  leftHasOrdinalFilter =  basisValuesLeft.basisValues().ordinalFilter().extent_int(0) > 0;
  const bool rightHasOrdinalFilter = basisValuesRight.basisValues().ordinalFilter().extent_int(0) > 0;
  TEUCHOS_TEST_FOR_EXCEPTION(leftHasOrdinalFilter || rightHasOrdinalFilter, std::invalid_argument, "Ordinal filters for BasisValues are not yet supported by PAMatrix");
  
  const int spaceDim = basisValuesLeft.spaceDim();
  
  // MARK: checks for supported construction
  INTREPID2_TEST_FOR_EXCEPTION(basisValuesLeft.spaceDim() != basisValuesRight.spaceDim(), std::invalid_argument, "basisValuesLeft and basisValuesRight must agree on the space dimension");
  
  const int leftFamilyCount  =  basisValuesLeft.vectorData().numFamilies();
  const int rightFamilyCount = basisValuesRight.vectorData().numFamilies();
  
  // we require that the number of tensor components in the vectors is the same for each vector entry
  // this is not strictly necessary, but it makes implementation easier, and we don't at present anticipate other use cases
  int numTensorComponentsLeft = -1;
  const bool leftIsVectorValued = basisValuesLeft.vectorData().isValid();
  
  if (leftIsVectorValued)
  {
    const auto &refVectorLeft   = basisValuesLeft.vectorData();
    int numFamiliesLeft         = refVectorLeft.numFamilies();
    int numVectorComponentsLeft = refVectorLeft.numComponents();
    Kokkos::Array<int,7> maxFieldsForComponentLeft  {0,0,0,0,0,0,0};
    for (int familyOrdinal=0; familyOrdinal<numFamiliesLeft; familyOrdinal++)
    {
      for (int vectorComponent=0; vectorComponent<numVectorComponentsLeft; vectorComponent++)
      {
        const TensorData<Scalar,DeviceType> &tensorData = refVectorLeft.getComponent(familyOrdinal,vectorComponent);
        if (tensorData.numTensorComponents() > 0)
        {
          if (numTensorComponentsLeft == -1)
          {
            numTensorComponentsLeft = tensorData.numTensorComponents();
          }
          INTREPID2_TEST_FOR_EXCEPTION(numVectorComponentsLeft != tensorData.numTensorComponents(), std::invalid_argument, "Each valid entry in basisValuesLeft must have the same number of tensor components as every other");
          for (int r=0; r<numTensorComponentsLeft; r++)
          {
            maxFieldsForComponentLeft[r] = std::max(tensorData.getTensorComponent(r).extent_int(0), maxFieldsForComponentLeft[r]);
          }
        }
      }
    }
  }
  else
  {
    numTensorComponentsLeft = basisValuesLeft.basisValues().tensorData(0).numTensorComponents(); // family ordinal 0
    for (int familyOrdinal = 0; familyOrdinal < leftFamilyCount; familyOrdinal++)
    {
      INTREPID2_TEST_FOR_EXCEPTION(basisValuesLeft.basisValues().tensorData(familyOrdinal).numTensorComponents() != numTensorComponentsLeft, std::invalid_argument, "All families must match in the number of tensor components");
    }
  }
  int numTensorComponentsRight = -1;
  const bool rightIsVectorValued = basisValuesRight.vectorData().isValid();
  
  if (rightIsVectorValued)
  {
    const auto &refVectorRight   = basisValuesRight.vectorData();
    int numFamiliesRight         = refVectorRight.numFamilies();
    int numVectorComponentsRight = refVectorRight.numComponents();
    Kokkos::Array<int,7> maxFieldsForComponentRight {0,0,0,0,0,0,0};
    for (int familyOrdinal=0; familyOrdinal<numFamiliesRight; familyOrdinal++)
    {
      for (int vectorComponent=0; vectorComponent<numVectorComponentsRight; vectorComponent++)
      {
        const auto &tensorData = refVectorRight.getComponent(familyOrdinal,vectorComponent);
        if (tensorData.numTensorComponents() > 0)
        {
          if (numTensorComponentsRight == -1)
          {
            numTensorComponentsRight = tensorData.numTensorComponents();
          }
          INTREPID2_TEST_FOR_EXCEPTION(numVectorComponentsRight != tensorData.numTensorComponents(), std::invalid_argument, "Each valid entry in basisValuesRight must have the same number of tensor components as every other");
          for (int r=0; r<numTensorComponentsRight; r++)
          {
            maxFieldsForComponentRight[r] = std::max(tensorData.getTensorComponent(r).extent_int(0), maxFieldsForComponentRight[r]);
          }
        }
      }
    }
    INTREPID2_TEST_FOR_EXCEPTION(numTensorComponentsRight != numTensorComponentsLeft, std::invalid_argument, "Right families must match left in the number of tensor components");
  }
  else
  {
    // check that right tensor component count agrees with left
    for (int familyOrdinal=0; familyOrdinal< rightFamilyCount; familyOrdinal++)
    {
      INTREPID2_TEST_FOR_EXCEPTION(basisValuesRight.basisValues().tensorData(familyOrdinal).numTensorComponents() != numTensorComponentsLeft, std::invalid_argument, "Right families must match left in the number of tensor components");
    }
  }
  const int numPointTensorComponents = cellMeasures.numTensorComponents() - 1;
    
  // MARK: check for separability
  if ((numPointTensorComponents == numTensorComponentsLeft) && basisValuesLeft.axisAligned() && basisValuesRight.axisAligned())
  {
    _separable = true;
  }
  else // general case (not axis-aligned + affine tensor-product structure)
  {
    _separable = false;
    // MARK: prepare composed transformation matrices
    const Data<Scalar,DeviceType> & leftTransform  = basisValuesLeft.transform();
    const Data<Scalar,DeviceType> & rightTransform = basisValuesRight.transform();
    const bool transposeLeft  = true;
    const bool transposeRight = false;
    //    auto timer = Teuchos::TimeMonitor::getNewTimer("mat-mat");
    //    timer->start();
    // transforms can be matrices -- (C,P,D,D): rank 4 -- or scalar weights -- (C,P): rank 2 -- or vector weights -- (C,P,D): rank 3
    Data<Scalar,DeviceType> composedTransform;
    // invalid/empty transforms are used when the identity is intended.
    const int leftRank  = leftTransform.rank();
    const int rightRank = rightTransform.rank();
    
    if (leftTransform.isValid() && rightTransform.isValid())
    {
      const bool bothRank4 = (leftRank == 4) && (rightRank == 4);
      const bool bothRank3 = (leftRank == 3) && (rightRank == 3);
      const bool bothRank2 = (leftRank == 2) && (rightRank == 2);
      const bool ranks32   = ((leftRank == 3) && (rightRank == 2)) || ((leftRank == 2) && (rightRank == 3));
      const bool ranks42   = ((leftRank == 4) && (rightRank == 2)) || ((leftRank == 2) && (rightRank == 4));
      
      if (bothRank4) // (C,P,D,D)
      {
        composedTransform = Data<Scalar,DeviceType>::allocateMatMatResult(transposeLeft, leftTransform, transposeRight, rightTransform);
        composedTransform.storeMatMat(transposeLeft, leftTransform, transposeRight, rightTransform);
      }
      else if (bothRank3) // (C,P,D)
      {
        // re-cast leftTransform as a rank 4 (C,P,1,D) object -- a 1 x D matrix at each (C,P).
        const int newRank   = 4;
        auto extents        = leftTransform.getExtents();
        auto variationTypes = leftTransform.getVariationTypes();
        extents[3]               = extents[2];
        extents[2]               = 1;
        variationTypes[3]        = variationTypes[2];
        variationTypes[2]        = CONSTANT;
        auto leftTransformMatrix = leftTransform.shallowCopy(newRank, extents, variationTypes);
        
        // re-cast rightTransform as a rank 4 (C,P,1,D) object -- a 1 x D matrix at each (C,P)
        extents                  = rightTransform.getExtents();
        variationTypes           = rightTransform.getVariationTypes();
        extents[3]               = extents[2];
        extents[2]               = 1;
        variationTypes[3]        = variationTypes[2];
        variationTypes[2]        = CONSTANT;
        auto rightTransformMatrix = rightTransform.shallowCopy(newRank, extents, variationTypes);
        
        composedTransform = Data<Scalar,DeviceType>::allocateMatMatResult(transposeLeft, leftTransformMatrix, transposeRight, rightTransformMatrix); // false: don't transpose
        composedTransform.storeMatMat(transposeLeft, leftTransformMatrix, transposeRight, rightTransformMatrix);
      }
      else if (bothRank2)
      {
        composedTransform = leftTransform.allocateInPlaceCombinationResult(leftTransform, rightTransform);
        composedTransform.storeInPlaceProduct(leftTransform, rightTransform);
        
        // re-cast composedTranform as a rank 4 (C,P,1,1) object -- a 1 x 1 matrix at each (C,P).
        const int newRank   = 4;
        auto extents        = composedTransform.getExtents();
        auto variationTypes = composedTransform.getVariationTypes();
        composedTransform = composedTransform.shallowCopy(newRank, extents, variationTypes);
      }
      else if (ranks32) // rank 2 / rank 3 combination.
      {
        const auto & rank3Transform = (leftRank == 3) ? leftTransform : rightTransform;
        const auto & rank2Transform = (leftRank == 2) ? leftTransform : rightTransform;
        
        composedTransform = DataTools::multiplyByCPWeights(rank3Transform, rank2Transform);
        
        // re-cast composedTransform as a rank 4 object:
        // logically, the original rank-3 transform can be understood as a 1xD matrix.  The composed transform is leftTransform^T * rightTransform, so:
        // - if left  has the rank-3 transform, composedTransform should be a (C,P,D,1) object -- a D x 1 matrix at each (C,P).
        // - if right has the rank-3 transform, composedTransform should be a (C,P,1,D) object -- a 1 x D matrix at each (C,P).
        const int newRank   = 4;
        auto extents        = composedTransform.getExtents();
        auto variationTypes = composedTransform.getVariationTypes();
        if (leftRank == 3)
        {
          // extents[3] and variationTypes[3] will already be 1 and CONSTANT, respectively
          // extents[3]               = 1;
          // variationTypes[3]        = CONSTANT;
        }
        else
        {
          extents[3]               = extents[2];
          extents[2]               = 1;
          variationTypes[3]        = variationTypes[2];
          variationTypes[2]        = CONSTANT;
        }
        composedTransform = composedTransform.shallowCopy(newRank, extents, variationTypes);
      }
      else if (ranks42) // rank 4 / rank 2 combination.
      {
        if (leftRank == 4)
        {
          // want to transpose left matrix, and multiply by the values from rightTransform
          // start with the multiplication:
          auto composedTransformTransposed = DataTools::multiplyByCPWeights(leftTransform, rightTransform);
          composedTransform = DataTools::transposeMatrix(composedTransformTransposed);
        }
        else // (leftRank == 2)
        {
          composedTransform = DataTools::multiplyByCPWeights(rightTransform, leftTransform);
        }
      }
      else
      {
        INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported transform combination");
      }
    }
    else if (leftTransform.isValid())
    {
      // rightTransform is the identity
      switch (leftRank)
      {
        case 4: composedTransform = DataTools::transposeMatrix(leftTransform); break;
        case 3:
        {
          // - if left  has the rank-3 transform, composedTransform should be a (C,P,D,1) object -- a D x 1 matrix at each (C,P).
          const int newRank   = 4;
          auto extents        = leftTransform.getExtents();
          auto variationTypes = leftTransform.getVariationTypes();
          
          composedTransform = leftTransform.shallowCopy(newRank, extents, variationTypes);
        }
          break;
        case 2: composedTransform = leftTransform; break;
        default:
          INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported transform combination");
      }
    }
    else if (rightTransform.isValid())
    {
      // leftTransform is the identity
      composedTransform = rightTransform;
      switch (rightRank)
      {
        case 4: composedTransform = rightTransform; break;
        case 3:
        {
          // - if right has the rank-3 transform, composedTransform should be a (C,P,1,D) object -- a 1 x D matrix at each (C,P).
          const int newRank   = 4;
          auto extents        = rightTransform.getExtents();
          auto variationTypes = rightTransform.getVariationTypes();
          extents[3]          = extents[2];
          variationTypes[3]   = variationTypes[2];
          extents[2]          = 1;
          variationTypes[2]   = CONSTANT;
          
          composedTransform = rightTransform.shallowCopy(newRank, extents, variationTypes);
        }
          break;
        case 2: composedTransform = rightTransform; break;
        default:
          INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported transform combination");
      }
    }
    else
    {
      // both left and right transforms are identity
      Kokkos::Array<ordinal_type,4> extents {basisValuesLeft.numCells(),basisValuesLeft.numPoints(),spaceDim,spaceDim};
      Kokkos::Array<DataVariationType,4> variationTypes {CONSTANT,CONSTANT,BLOCK_PLUS_DIAGONAL,BLOCK_PLUS_DIAGONAL};
      
      Kokkos::View<Scalar*,DeviceType> identityUnderlyingView("Intrepid2::FST::integrate() - identity view",spaceDim);
      Kokkos::deep_copy(identityUnderlyingView, 1.0);
      composedTransform = Data<Scalar,DeviceType>(identityUnderlyingView,extents,variationTypes);
    }
    // allocate weighted transform
    _composedWeightedTransform = Impl::allocateComposedWeightedTransform<DeviceType,Scalar>(composedTransform,cellMeasures);
    auto composedWeightedTransform = _composedWeightedTransform; // avoid implicit reference to this
    // MARK: fill weighted transform container
    int rank = composedWeightedTransform.rank();
    int cellDataExtent    = composedWeightedTransform.getDataExtent(0);
    int numPoints         = composedWeightedTransform.getDataExtent(1);
    int d1_dim            = composedWeightedTransform.getDataExtent(2);
    int d2_dim            = composedWeightedTransform.getDataExtent(3);
    auto d1_variationType = composedWeightedTransform.getVariationTypes()[2];
    
    if (rank == 2)
    {
      Kokkos::Array<int,2> lowerBounds {0,0};
      Kokkos::Array<int,2> upperBounds {cellDataExtent,numPoints};
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>(lowerBounds, upperBounds);
      
      Kokkos::parallel_for("compute weighted transform", policy,
                           KOKKOS_LAMBDA (const int &cellDataOrdinal, const int &pointOrdinal) {
        const Scalar & w = cellMeasures(cellDataOrdinal, pointOrdinal);
        Scalar & result  = composedWeightedTransform.getWritableEntry(cellDataOrdinal,pointOrdinal);
        result = w * composedTransform(cellDataOrdinal,pointOrdinal);
      });
    }
    else if ((rank == 3) || ((rank == 4) && (d1_variationType == BLOCK_PLUS_DIAGONAL)))
    {
      Kokkos::Array<int,3> lowerBounds {0,0,0};
      Kokkos::Array<int,3> upperBounds {cellDataExtent,numPoints,d1_dim};
      bool passThroughMatrixDims = (d1_variationType == BLOCK_PLUS_DIAGONAL); // if BLOCK_PLUS_DIAGONAL, it's a matrix, but everything is packed into the D1 dimension, and we want to sidestep the logic that tries to compute the matrix entry index based on (d1,d2) arguments.
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>(lowerBounds, upperBounds);
      
      Kokkos::parallel_for("compute weighted transform", policy,
                           KOKKOS_LAMBDA (const int &cellDataOrdinal, const int &pointOrdinal, const int &d1) {
        const Scalar & w = cellMeasures(cellDataOrdinal, pointOrdinal);
        Scalar & result  = composedWeightedTransform.getWritableEntryWithPassThroughOption(passThroughMatrixDims,cellDataOrdinal,pointOrdinal,d1);
        result = w * composedTransform(cellDataOrdinal,pointOrdinal,d1);
      });
    }
    else if (rank == 4)
    {
      Kokkos::Array<int,4> lowerBounds {0,0,0,0};
      Kokkos::Array<int,4> upperBounds {cellDataExtent,numPoints,d1_dim,d2_dim};
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<4>>(lowerBounds, upperBounds);
      
      Kokkos::parallel_for("compute weighted transform", policy,
                           KOKKOS_LAMBDA (const int &cellDataOrdinal, const int &pointOrdinal, const int &d1, const int &d2) {
        const Scalar & w = cellMeasures(cellDataOrdinal, pointOrdinal);
        Scalar & result  = composedWeightedTransform.getWritableEntry(cellDataOrdinal,pointOrdinal,d1,d2);
        result = w * composedTransform(cellDataOrdinal,pointOrdinal,d1,d2);
      });
    }
  }
} // PAMatrix()

template<typename DeviceType,class Scalar>
PAMatrix<DeviceType,Scalar>::PAMatrix(const TransformedBasisValues<Scalar,DeviceType> basisValues,
                                      const TensorData<Scalar,DeviceType> cellMeasures,
                                      const ScalarView<Orientation,DeviceType> orientations)
:
PAMatrix<DeviceType,Scalar>(basisValues,cellMeasures,basisValues,orientations)
{}

template<typename DeviceType,class Scalar>
Data<Scalar,DeviceType> PAMatrix<DeviceType,Scalar>::allocateMatrixStorage()
{
  // Allocates a (C,F,F) container for storing integral data
  
  // Ordinal filter is used for Serendipity basis; we don't yet support Serendipity for PAMatrix.
  // (When we do, the strategy will likely be to apply the right filter at the "middle" of the operator sequence, and the left filter at the end.  This does mean that the intermediate containers for the right operators will be sized for the unfiltered basis; the intermediate containers for the left operators will be sized like unfiltered left x filtered right.)
  const bool  leftHasOrdinalFilter =  _basisValuesLeft.basisValues().ordinalFilter().extent_int(0) > 0;
  const bool rightHasOrdinalFilter = _basisValuesRight.basisValues().ordinalFilter().extent_int(0) > 0;
  TEUCHOS_TEST_FOR_EXCEPTION(leftHasOrdinalFilter || rightHasOrdinalFilter, std::invalid_argument, "Ordinal filters for BasisValues are not yet supported by PAMatrix");
  
  // determine cellDataExtent and variation type.  We currently support CONSTANT, MODULAR, and GENERAL as possible output variation types, depending on the inputs.
  // If cellMeasures has non-trivial tensor structure, the rank-1 cell Data object is the first component.
  // If cellMeasures has trivial tensor structure, then the first and only component has the cell index in its first dimension.
  // I.e., either way the relevant Data object is cellMeasures.getTensorComponent(0)
  const int CELL_DIM = 0;
  const auto cellMeasureData = _cellMeasures.getTensorComponent(0);
  const auto leftTransform = _basisValuesLeft.transform();
  
  DimensionInfo combinedCellDimInfo = cellMeasureData.getDimensionInfo(CELL_DIM);
  // transforms may be invalid, indicating an identity transform.  If so, it will not constrain the output at all.
  if (_basisValuesLeft.transform().isValid())
  {
    combinedCellDimInfo = combinedDimensionInfo(combinedCellDimInfo, _basisValuesLeft.transform().getDimensionInfo(CELL_DIM));
  }
  if (_basisValuesRight.transform().isValid())
  {
    combinedCellDimInfo = combinedDimensionInfo(combinedCellDimInfo, _basisValuesRight.transform().getDimensionInfo(CELL_DIM));
  }

  DataVariationType cellVariationType = combinedCellDimInfo.variationType;
  int cellDataExtent                  = combinedCellDimInfo.dataExtent;
  
  const int numCells       = _basisValuesLeft.numCells();
  const int numFieldsLeft  = _basisValuesLeft.numFields();
  const int numFieldsRight = _basisValuesRight.numFields();
  
  Kokkos::Array<int,3> extents {numCells, numFieldsLeft, numFieldsRight};
  Kokkos::Array<DataVariationType,3> variationTypes {cellVariationType,GENERAL,GENERAL};
  
  if (cellVariationType != CONSTANT)
  {
    Kokkos::View<Scalar***,DeviceType> data("Intrepid2::PAMatrix matrix storage",cellDataExtent,numFieldsLeft,numFieldsRight);
    return Data<Scalar,DeviceType>(data, extents, variationTypes);
  }
  else
  {
    Kokkos::View<Scalar**,DeviceType> data("Intrepid2::PAMatrix matrix storage",numFieldsLeft,numFieldsRight);
    return Data<Scalar,DeviceType>(data, extents, variationTypes);
  }
} // allocateMatrixStorage()

template<typename DeviceType,class Scalar>
void PAMatrix<DeviceType,Scalar>::apply(const ScalarView<Scalar,DeviceType> &outputVector,
                                        const ScalarView<Scalar,DeviceType> & inputVector)
{
  // TODO: implement this
}

template<typename DeviceType,class Scalar>
void PAMatrix<DeviceType,Scalar>::assemble(Data<Scalar,DeviceType> &integrals)
{
  //placeholder implementation: just invoke IntegrationTools
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace    = typename DeviceType::memory_space;
  
  bool sumInto = false;
  double approximateFlopCountIntegrate = 0;
  IntegrationTools<DeviceType>::integrate(integrals, _basisValuesLeft, _cellMeasures, _basisValuesRight, sumInto, &approximateFlopCountIntegrate);
  ExecutionSpace().fence();
  
  auto leftBasis  =  _basisValuesLeft.basisValues().getBasis();
  auto rightBasis = _basisValuesRight.basisValues().getBasis();
  
  if (_orientations.size() > 0)
  {
    // modify integrals by orientations -- we are NOT allowed to use the same view as source and result, so let's create a mirror view for source.
    auto unorientatedValues = Kokkos::create_mirror_view_and_copy(MemorySpace(), integrals.getUnderlyingView());
    OrientationTools<DeviceType>::modifyMatrixByOrientation(integrals.getUnderlyingView(), unorientatedValues,
                                                            _orientations, leftBasis.get(), rightBasis.get());
    ExecutionSpace().fence();
  }
}

} // end namespace Intrepid2
#endif
