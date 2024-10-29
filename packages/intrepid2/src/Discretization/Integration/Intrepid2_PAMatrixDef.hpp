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
#include "Intrepid2_IntegrationTools.hpp"
#include "Intrepid2_OrientationTools.hpp"

//#ifdef __APPLE__
//#include <Accelerate/Accelerate.h>
//#else
//#include <cblas.h>
//#endif

#include <Teuchos_BLAS.hpp>

#ifdef HAVE_INTREPID2_KOKKOSKERNELS
#include <KokkosBlas.hpp>
#endif

namespace Intrepid2 {

namespace Impl
{
//! For matrix-valued A(C,P,Da,Db) and vector-valued B(C,P,Db), output C(C,P,Da) representing the pointwise matrix-vector product.  At present, on the assumption that Da and Db are small, potentially unit-valued, we compute this in a KOKKOS_LAMBDA, but we may want to invoke GEMM for the multiply in the future.
template<typename DeviceType, class Scalar>
void pointDataMultiply(const ordinal_type numCells, const ordinal_type numPoints, const ordinal_type aSpan, const ordinal_type bSpan,
                       const Scalar* A, const Scalar *B, Scalar *C)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,numPoints,aSpan});
  
  Kokkos::parallel_for("pointDataMultiply", policy,
                       KOKKOS_LAMBDA(const ordinal_type &cell, const ordinal_type &point, const ordinal_type &a)
  {
    Scalar value = 0;
    const Scalar* entryA = A + (a + (point + cell * numPoints) * aSpan) * bSpan;
    const Scalar* entryB = B + (point + cell * numPoints) * bSpan;
    for (int b=0; b<bSpan; b++)
    {
      value += *entryA * *entryB;
      entryA++;
      entryB++;
    }
    Scalar* entryC = C + a + (point + cell * numPoints) * aSpan;
    *entryC = value;
  });
  ExecutionSpace().fence();
}

  template<typename DeviceType,typename Scalar>
  std::enable_if_t<std::is_same<typename DeviceType::execution_space, typename Kokkos::Serial::execution_space>::value>
  gemm(const char transA, const char transB,
       const ordinal_type &M, const ordinal_type &N, const ordinal_type &K,
       const Scalar &alpha, const Scalar* A, const ordinal_type &LDA,
       const Scalar *B, const Scalar &beta, Scalar *C)
  {
    Teuchos::ETransp trA = (transA == 'T') ? Teuchos::TRANS : (transA == 'C') ? Teuchos::CONJ_TRANS : Teuchos::NO_TRANS;
    Teuchos::ETransp trB = (transB == 'T') ? Teuchos::TRANS : (transB == 'C') ? Teuchos::CONJ_TRANS : Teuchos::NO_TRANS;
    Teuchos::BLAS<int,Scalar> blas;
    const ordinal_type LDB = K;
    const ordinal_type LDC = M;
    blas.GEMM(trA, trB, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
  }

#ifdef HAVE_INTREPID2_KOKKOSKERNELS
  template<typename DeviceType,typename Scalar>
  std::enable_if_t<!std::is_same<typename DeviceType::execution_space, typename Kokkos::Serial::execution_space>::value>
  gemm(const char transA, const char transB,
       const ordinal_type &M, const ordinal_type &N, const ordinal_type &K,
       const Scalar &alpha, const Scalar* A, const ordinal_type &LDA,
       const Scalar *B, const Scalar &beta, Scalar *C)
  {
    using ConstView2D = Kokkos::View<const Scalar**, DeviceType, Kokkos::MemoryUnmanaged>;
    using      View2D = Kokkos::View<      Scalar**, DeviceType, Kokkos::MemoryUnmanaged>;
    ConstView2D AView(A,M,K);
    ConstView2D BView(B,N,K);
         View2D CView(C,M,N);
    
    typename DeviceType::execution_space exec_space;
    KokkosBlas::gemm(exec_space, &transA, &transB, alpha, AView, BView, beta, CView);
  }
#else
  template<typename DeviceType,typename Scalar>
  std::enable_if_t<!std::is_same<typename DeviceType::execution_space, typename Kokkos::Serial::execution_space>::value>
  gemm(const char transA, const char transB,
       const ordinal_type &M, const ordinal_type &N, const ordinal_type &K,
       const Scalar &alpha, const Scalar* A, const ordinal_type &LDA,
       const Scalar *B, const Scalar &beta, Scalar *C)
  {
    INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "To support gemm on this ExecutionSpace, please build Intrepid2 with KokkosKernels");
  }
#endif

// Define GemmDeviceType: use Kokkos-supported GPUs if enabled; otherwise use serial.  Note that on macOS if you use serial and are using Apple's BLAS on an M-series Mac, it will run on the M-series GPU (and will be very fast).
#if defined(KOKKOS_ENABLE_CUDA)
using GemmDeviceType = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
using GemmDeviceType = Kokkos::HIP;
#else
using GemmDeviceType = Kokkos::Serial;
#endif

/*!
 Given tensor data with shape (D1, D2, …, Dn), prepare for a contraction in the Dk dimension by reordering as
  (Di, D1, D2, …, D{k-1}, D{k+1}, …, Dn)
 */
  template<typename DeviceType,class Scalar>
  class TensorReorderForGemmFunctor
  {
  public:
    using ExecutionSpace = typename DeviceType::execution_space;
    using View1D = Kokkos::View<Scalar*,DeviceType>;
    
    View1D outputView_;
    View1D  inputView_;
    
    int  leftDims_ = 1; // product D1 * D2 … * D{k-1}
    int      kDim_ = 1; // Dk
    int rightDims_ = 1; // product D{k+1} * … * Dn
    
    static constexpr bool layoutLeft_ = true; // aka column-major (Fortran-style): columns are together
    
    //! outputView and inputView must each be large enough to accommodate leftDims * iDim * rightDims, but they may be oversized.
    TensorReorderForGemmFunctor(View1D outputView, View1D inputView,
                                int leftDims, int kDim, int rightDims)
    :
    outputView_(outputView),
    inputView_(inputView),
    leftDims_(leftDims),
    kDim_(kDim),
    rightDims_(rightDims)
    {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()( const int &i, const int &j, const int &k) const
    {
      // source has (i,k,j)
      // dest   has (k,i,j)
      
      // i should iterate over leftDims (flattened)
      // j should iterate over rightDims (flattened)
      // k should iterate over Dk
      if (layoutLeft_) // column-major
      {
        const int dest_idx = j + (i + k * leftDims_) * rightDims_;
        const int  src_idx = j + (k + i * kDim_    ) * rightDims_;
        outputView_(dest_idx) = inputView_(src_idx);
      }
      else
      {
        const int dest_idx = k + (i + j * leftDims_) * kDim_;
        const int  src_idx = i + (k + j * kDim_    ) * leftDims_;
        outputView_(dest_idx) = inputView_(src_idx);
      }
    }
    
    void run()
    {
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{leftDims_,rightDims_,kDim_});
      Kokkos::parallel_for("PAMatrix: tensor reorder for gemm", policy, *this);
    }
  };

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

// blas.GEMM(trA, trB, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

  //! take an M x K matrix A and contract with an N1 x K x N2 tensor B to produce an N1 x M x N2 output C.
  //! This is done in terms of a series of constituent gemms, iterating over the n2 dimension.  We launch these in a Kokkos::parallel_for on the
  //! *host* execution space.  This allows us to invoke a synchronous gemm call in an asynchronous way.  In particular, on macOS, Apple's Accelerate
  //! framework provides a gemm implementation that invokes the GPU on M-series processors, but this waits for completion before it returns, and
  //! typically does not saturate the GPU.  If Kokkos is built with OpenMP support, we can thus increase parallelism by however many OpenMP threads
  //! are available.  Similar considerations apply to KokkosKernels's gemm implementation on CUDA or HIP DeviceType.  We do need to be careful not to
  //! launch a KokkosKernels gemm under OpenMP with an OpenMP DispatchExecutionSpace: the basic rule here is that GemmDeviceType must
//! be different from DispatchExecutionSpace unless they are both Serial.
  template<typename GemmDeviceType, class Scalar, typename DispatchExecutionSpace=Kokkos::DefaultHostExecutionSpace>
  std::enable_if_t<
    !std::is_same<DispatchExecutionSpace, GemmDeviceType>::value ||
    (std::is_same<DispatchExecutionSpace, Kokkos::Serial>::value && std::is_same<GemmDeviceType, Kokkos::Serial>::value)
  >
  matrixTensorContractionLayoutLeft(const ordinal_type &M, const ordinal_type &N1, const ordinal_type &N2, const ordinal_type &K,
                                    const Scalar &alpha, const Scalar* A, const ordinal_type &LDA,
                                    const Scalar *B,
                                    const Scalar &beta, Scalar *C)
  {
    // we assume layout left, so that the B tensor (i,k,j) index flattens to j + k * LDB + i * K * N2.
    // this means that the slice B(i,:,:) is a K x N2 matrix, contiguous in memory, at offset i * K * N2.
    // similarly, the slice C(i,:,:) is a M x N2 matrix, contiguous in memory, at offset i * K * N2.
    
    auto policy = Kokkos::RangePolicy<DispatchExecutionSpace>(0,N1);
    
    const ordinal_type KN2 = K * N2;
    Kokkos::parallel_for("matrixTensorContractionLayoutLeft: GEMM dispatch", policy,
                         KOKKOS_LAMBDA(const ordinal_type &i)
    {
      const ordinal_type i_offset = i * KN2;
      const auto B_i = B + i_offset;
      const auto C_i = C + i_offset;
      gemm<GemmDeviceType>('N', 'N', M, N2, K, alpha, A, LDA, B_i, beta, C_i);
    });
    
    DispatchExecutionSpace().fence();
  }

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

  const bool layoutLeft = layoutLeft_;
  
  const bool  leftHasOrdinalFilter =  basisValuesLeft.basisValues().ordinalFilter().extent_int(0) > 0;
  const bool rightHasOrdinalFilter = basisValuesRight.basisValues().ordinalFilter().extent_int(0) > 0;
  TEUCHOS_TEST_FOR_EXCEPTION(leftHasOrdinalFilter || rightHasOrdinalFilter, std::invalid_argument, "Ordinal filters for BasisValues are not yet supported by PAMatrix");
  
  const int spaceDim = basisValuesLeft.spaceDim();
  
  // MARK: checks for supported construction
  INTREPID2_TEST_FOR_EXCEPTION(basisValuesLeft.spaceDim() != basisValuesRight.spaceDim(), std::invalid_argument, "basisValuesLeft and basisValuesRight must agree on the space dimension");
  
  const int leftFamilyCount  =  basisValuesLeft.basisValues().numFamilies();
  const int rightFamilyCount = basisValuesRight.basisValues().numFamilies();
  
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
  
  // MARK: Set up component integrations
  const int leftComponentCount  = leftIsVectorValued ? basisValuesLeft. vectorData().numComponents() : 1;
  const int rightComponentCount = rightIsVectorValued ? basisValuesRight.vectorData().numComponents() : 1;
  
  int leftFieldOrdinalOffset = 0; // keeps track of the number of fields in prior families
  for (int leftFamilyOrdinal=0; leftFamilyOrdinal<leftFamilyCount; leftFamilyOrdinal++)
  {
    // "a" keeps track of the spatial dimension over which we are integrating in the left vector.
    // Components are allowed to span several dimensions; we keep track of the offset for the component in a_offset
    int a_offset = 0;
    bool haveLaunchedContributionToCurrentFamilyLeft = false; // helps to track whether we need a Kokkos::fence before launching a kernel.
    for (int leftComponentOrdinal=0; leftComponentOrdinal<leftComponentCount; leftComponentOrdinal++)
    {
      TensorData<Scalar,DeviceType> leftComponent = leftIsVectorValued ? basisValuesLeft.vectorData().getComponent(leftFamilyOrdinal, leftComponentOrdinal)
                                                                       : basisValuesLeft.basisValues().tensorData(leftFamilyOrdinal);
      if (!leftComponent.isValid())
      {
         // represents zero
        a_offset += basisValuesLeft.vectorData().numDimsForComponent(leftComponentOrdinal);
        continue;
      }
      // set up the individual operators as 1D views
      // left operators contract in the point (and space) dimensions
      std::vector<OpSpec> leftOperators;
      for (int r=0; r<leftComponent.numTensorComponents(); r++)
      {
        const auto opData  = leftComponent.getTensorComponent(r).getUnderlyingView();
        const int opFields = opData.extent_int(0);
        const int opPoints = opData.extent_int(1);
        View1D opView("leftOp 1D view", opData.size());
        if (opData.rank() == 2) // (F,P)
        {
          auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{opFields,opPoints});
          Kokkos::parallel_for("pack 1D opView", policy,
          KOKKOS_LAMBDA(const int &field, const int &pt)
          {
            const int idx = layoutLeft ? pt + field * opPoints : field + pt * opFields;
            opView(idx) = opData(field,pt);
          });
        }
        else if (opData.rank() == 3) // (F,P,D)
        {
          const int opDim = opData.extent_int(2);
          auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{opFields,opPoints,opDim});
          Kokkos::parallel_for("pack 1D opView", policy,
          KOKKOS_LAMBDA(const int &field, const int &pt, const int &d)
          {
            const int idx = layoutLeft ? d + (pt + field * opPoints) * opDim : field + (pt + d * opPoints) * opFields;
            opView(idx) = opData(field,pt,d);
          });
        }
        else
        {
          INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "PAMatrix: Unsupported component operator rank");
        }
        leftOperators.push_back({opView,opFields,opPoints});
      }
      
      int rightFieldOrdinalOffset = 0; // keeps track of the number of fields in prior families // TODO: figure out what a nonzero value means for matrix-free apply() implementation
      for (int rightFamilyOrdinal=0; rightFamilyOrdinal<rightFamilyCount; rightFamilyOrdinal++)
      {
        // "b" keeps track of the spatial dimension over which we are integrating in the right vector
        // components are allowed to span several dimensions; we keep track of the offset for the component in b_offset
        bool haveLaunchedContributionToCurrentFamilyRight = false; // helps to track whether we need a Kokkos::fence before launching a kernel.
        int b_offset = 0;
        for (int rightComponentOrdinal=0; rightComponentOrdinal<rightComponentCount; rightComponentOrdinal++)
        {
          TensorData<Scalar,DeviceType> rightComponent =
             rightIsVectorValued ? basisValuesRight.vectorData().getComponent(rightFamilyOrdinal, rightComponentOrdinal)
                                 : basisValuesRight.basisValues().tensorData(rightFamilyOrdinal);
          if (!rightComponent.isValid())
          {
             // represents zero
            b_offset += basisValuesRight.vectorData().numDimsForComponent(rightComponentOrdinal);
            continue;
          }
          
          INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(leftComponent.numTensorComponents() != rightComponent.numTensorComponents(), std::invalid_argument, "left TensorData and right TensorData have different number of tensor components.  This is not supported.");
          
          // right operators contract in the field dimension
          std::vector<OpSpec> rightOperators;
          for (int r=0; r<rightComponent.numTensorComponents(); r++)
          {
            const auto  opData = rightComponent.getTensorComponent(r).getUnderlyingView();
            const int opFields = opData.extent_int(0);
            const int opPoints = opData.extent_int(1);
            
            View1D opView("rightOp 1D view", opData.size());
            if (opData.rank() == 2) // (F,P), but will pack as (P,F)
            {
              auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{opFields,opPoints});
              Kokkos::parallel_for("pack 1D opView", policy,
              KOKKOS_LAMBDA(const int &field, const int &pt)
              {
                const int idx = layoutLeft ? field + pt * opFields : pt + field * opPoints;
                opView(idx) = opData(field,pt);
              });
            }
            else if (opData.rank() == 3) // (F,P,D), but will pack as (P,D,F) for contraction in F
            {
              const int opDim = opData.extent_int(2);
              auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{opFields,opPoints,opDim});
              Kokkos::parallel_for("pack 1D opView", policy,
              KOKKOS_LAMBDA(const int &field, const int &pt, const int &d)
              {
                const int idx = layoutLeft ? field + (pt + d * opPoints) * opFields : d + (pt + field * opPoints) * opDim ;
                opView(idx) = opData(field,pt,d);
              });
            }
            else
            {
              INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "PAMatrix: Unsupported component operator rank");
            }
            rightOperators.push_back({opView,opPoints,opFields});
          }
          
          const int aSpan =  leftComponent.extent_int(2);
          const int bSpan = rightComponent.extent_int(2);
          
          const int numCells      = _composedWeightedTransform.extent_int(0);
          const int numPoints     = _composedWeightedTransform.extent_int(1);
          
          PointDataSpec pointDataSpec{numCells, numPoints, a_offset, b_offset, aSpan, bSpan};
          
          if (_pointDataCache.find(pointDataSpec) == _pointDataCache.end())
          {
            const int pointDataSize = numCells * numPoints * aSpan * bSpan;
            View1D pointDataView("pointDataView", pointDataSize);
            
            auto composedWeightedTransform = _composedWeightedTransform;
            
            if (_composedWeightedTransform.rank() == 2) // (C,P): pointwise weight
            {
              auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{numCells,numPoints});
              Kokkos::parallel_for("pack 1D pointData", policy,
                                   KOKKOS_LAMBDA(const int &cell, const int &pt)
                                   {
                const int idx = layoutLeft ? cell + pt * numCells : pt + cell * numPoints ;
                pointDataView(idx) = composedWeightedTransform(cell,pt);
              });
            }
            else if (_composedWeightedTransform.rank() == 3) // (C,P,D): contract in b or expand in a
            {
              const bool contraction = (bSpan > 1);
              const int dOffset = contraction ? b_offset : a_offset;
              const int dSpan   = contraction ?    bSpan : aSpan;
              auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,numPoints,dSpan});
              Kokkos::parallel_for("pack 1D pointData", policy,
                                   KOKKOS_LAMBDA(const int &cell, const int &pt, const int &d)
                                   {
                const int idx = layoutLeft ? cell + (pt + d * numPoints) * numCells : d + (pt + cell * numPoints) * dSpan;
                pointDataView(idx) = composedWeightedTransform(cell,pt,dOffset + d);
              });
            }
            else if (_composedWeightedTransform.rank() == 4) // (C,P,D,D): contract in b and expand in a at each point
            {
              auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<4>>({0,0,0,0},{numCells,numPoints,aSpan,bSpan});
              Kokkos::parallel_for("pack 1D pointData", policy,
                                   KOKKOS_LAMBDA(const int &cell, const int &pt, const int &da, const int &db)
                                   {
                const int idx = layoutLeft ? cell + (pt + (da + db * aSpan) * numPoints) * numCells
                : db + (da + (pt + cell * numPoints) * aSpan) * bSpan ;
                pointDataView(idx) = composedWeightedTransform(cell,pt,a_offset + da,b_offset + db);
              });
            }
            _pointDataCache[pointDataSpec] = pointDataView;
          }
          componentIntegralsToSum_.push_back({leftOperators,pointDataSpec,rightOperators});
          
          b_offset += rightIsVectorValued ? basisValuesRight.vectorData().numDimsForComponent(rightComponentOrdinal) : 1;
        }
        rightFieldOrdinalOffset += rightIsVectorValued ? basisValuesRight.vectorData().numFieldsInFamily(rightFamilyOrdinal) : basisValuesRight.basisValues().numFieldsInFamily(rightFamilyOrdinal);
      }
      a_offset += leftIsVectorValued ? basisValuesLeft.vectorData().numDimsForComponent(leftComponentOrdinal) : 1;
    }
    leftFieldOrdinalOffset += leftIsVectorValued ? basisValuesLeft.vectorData().numFieldsInFamily(leftFamilyOrdinal) : basisValuesLeft.basisValues().numFieldsInFamily(leftFamilyOrdinal);
  }
  
  // set maxIntermediateSize_: the per-cell size required for intermediate computations, which is used to size the workspaces
  const int F2 = basisValuesRight.extent_int(1); // C,F,P,…
  const int F1 = basisValuesLeft.extent_int(1);
  maxIntermediateSize_ = std::max(F1,F2);
  
  for (const auto &entry : componentIntegralsToSum_)
  {
    const auto & leftOps       = std::get<0>(entry);
    const auto & pointDataSpec = std::get<1>(entry);
    const auto & rightOps      = std::get<2>(entry);
    
    int perCellSize = F2; // num basis coefficients in the vector we multiply
    // we start the contraction on the right
    for (const auto &rightOp : rightOps)
    {
      perCellSize /= rightOp.N;
      perCellSize *= rightOp.M;
      maxIntermediateSize_ = max(perCellSize,maxIntermediateSize_);
    }
    
    perCellSize /= pointDataSpec.bSpan;
    perCellSize *= pointDataSpec.aSpan;
    maxIntermediateSize_ = max(perCellSize,maxIntermediateSize_);
    
    for (const auto &leftOp : leftOps)
    {
      perCellSize /= leftOp.N;
      perCellSize *= leftOp.M;
      maxIntermediateSize_ = max(perCellSize,maxIntermediateSize_);
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
Kokkos::View<Scalar*,DeviceType> PAMatrix<DeviceType,Scalar>::allocateWorkspace(const ordinal_type &worksetSize)
{
  using View1D = Kokkos::View<Scalar*,DeviceType>;
  const int size1D = maxIntermediateSize_ * worksetSize;
  return View1D("PAMatrix workspace", size1D);
}

template<typename DeviceType,class Scalar>
Kokkos::View<Scalar*,DeviceType> PAMatrix<DeviceType,Scalar>::allocateWorkspace(const ordinal_type &worksetSize,
                                                                                const ordinal_type &n)
{
  using View1D = Kokkos::View<Scalar*,DeviceType>;
  const int size1D = maxIntermediateSize_ * worksetSize * n;
  return View1D("PAMatrix workspace", size1D);
}

template<typename DeviceType,class Scalar>
void PAMatrix<DeviceType,Scalar>::apply(const ScalarView<Scalar,DeviceType> &outputVector,
                                        const ScalarView<Scalar,DeviceType> & inputVector,
                                        const Kokkos::View<Scalar*,DeviceType> &workspace1,
                                        const Kokkos::View<Scalar*,DeviceType> &workspace2)
{
  // TODO: add worksetSize argument (assume workSetSize == C for now)
  using ExecutionSpace = typename DeviceType::execution_space;
  using View1D = Kokkos::View<Scalar*,DeviceType>;
  
  const ordinal_type C  = inputVector.extent_int(0); // C, F2, N
  const ordinal_type F2 = inputVector.extent_int(1);
  const ordinal_type N  = inputVector.extent_int(2);
  const ordinal_type F1 = outputVector.extent_int(1); // C, F1, N
  
  const double alpha = 1.0;
  const double beta  = 0.0;
  
  Kokkos::deep_copy(outputVector, 0.0);
  
  // For vector-dot-vector integrals (e.g.), we need to integrate left x components against right x components, etc., and sum.
  // Each of these is one pass, and we accumulate in outputVector.
  const int numIntegrationPasses = int(componentIntegralsToSum_.size());
  
  // TODO: add loop over cell worksets (right now the below assumes worksetSize == C)
  for (int integrationPass=0; integrationPass<numIntegrationPasses; integrationPass++)
  {
    const auto &integral_tuple = componentIntegralsToSum_[integrationPass];
    // right integrals: replace F2j basis coefficients with evaluations at Pj
    const auto & rightIntegrals = std::get<2>(integral_tuple);
    int numRightIntegrals = int(rightIntegrals.size());
    
    const PointDataSpec & pointDataSpec = std::get<1>(integral_tuple);
    
    const auto & leftIntegrals = std::get<0>(integral_tuple);
    int numLeftIntegrals = int(leftIntegrals.size());
    
    // set workspace1 to the input data, with layout left
    auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{C,F2,N});
    Kokkos::parallel_for("PAMatrix::apply(): copy inputVector into workspace", policy,
    KOKKOS_LAMBDA(const ordinal_type &c, const ordinal_type &f, const ordinal_type &n)
    {
      const ordinal_type idx = n + (f + n * F2) * N;
      workspace1(idx) = inputVector(c,f,n);
    }
    );
    ExecutionSpace().fence();
    
    // the right integrals are ordered in the natural dimension ordering: x integrals come first.
    // this means that the first tensor contraction is (P_x,F2_x) against (C,F2_x,F2_y*…*F2_n*N)
    int N1 = C;
    int N2 = F2 * N; // will modify before first use, below
    for (int j=0; j<numRightIntegrals; j++)
    {
      // we alternate whether we are placing intermediate results in workspace1 or workspace2
      auto  in = (j%2 == 0) ? workspace1 : workspace2;
      auto out = (j%2 == 0) ? workspace2 : workspace1;
      
      auto op = rightIntegrals[j];
      // contraction of M x K with tensor of shape N1 x K x N2;
      const ordinal_type & M = op.M;
      const ordinal_type & K = op.N;
      
      const auto A = op.opView.data();
      const ordinal_type LDA = M; // will need to revise if we ever pad our operators (for byte alignment)
      N2 /= K;
      const auto B = in.data();
      auto C = out.data();
      Impl::matrixTensorContractionLayoutLeft<Impl::GemmDeviceType>(M, N1, N2, K, alpha, A, LDA, B, beta, C);
      N1 *= K;
    }
    auto  pointDataIn = (numRightIntegrals%2 == 0) ? workspace1 : workspace2; // pointwise result from contractions so far
    auto pointDataOut = (numRightIntegrals%2 == 0) ? workspace2 : workspace1; // pointwise output from weighting with pointData
    
    auto pointData = _pointDataCache[pointDataSpec]; // pointwise weights
    Impl::pointDataMultiply<DeviceType,Scalar>(pointDataSpec.C, pointDataSpec.P, pointDataSpec.aSpan, pointDataSpec.bSpan,
                                               pointData.data(), pointDataIn.data(), pointDataOut.data());
    
    // the left integrals are ordered in the natural dimension ordering: x integrals come first.
    // this means that the first tensor contraction is (F1_x,P_x) against (C,P_x,P_y*…*P_n*N)
    N1 = C;
    N2 = F1 * N; // will modify before first use, below
    for (int i=0; i<numLeftIntegrals; i++)
    {
      // we alternate whether we are placing intermediate results in workspace1 or workspace2
      auto  in = ((i+numRightIntegrals+1)%2 == 0) ? workspace1 : workspace2;
      auto out = ((i+numRightIntegrals+1)%2 == 0) ? workspace2 : workspace1;
      
      auto op = leftIntegrals[i];
      // contraction of M x K with tensor of shape N1 x K x N2;
      const ordinal_type & M = op.M;
      const ordinal_type & K = op.N;
      
      const auto A = op.opView.data();
      const ordinal_type LDA = M; // will need to revise if we ever pad our operators (for byte alignment)
      N2 /= K;
      const auto B = in.data();
      auto C = out.data();
      Impl::matrixTensorContractionLayoutLeft<Impl::GemmDeviceType>(M, N1, N2, K, alpha, A, LDA, B, beta, C);
      N1 *= K;
    }
    auto finalOut = ((numLeftIntegrals+numRightIntegrals+1)%2 == 0) ? workspace1 : workspace2;
    // Sum finalOut into outputVector
    policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{C,F1,N});
    Kokkos::parallel_for("PAMatrix::apply(): sum finalOut into outputVector", policy,
    KOKKOS_LAMBDA(const ordinal_type &c, const ordinal_type &f, const ordinal_type &n)
    {
      const ordinal_type idx = n + (f + n * F1) * N;
      outputVector(c,f,n) += finalOut(idx);
    });
    ExecutionSpace().fence();
  }
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
