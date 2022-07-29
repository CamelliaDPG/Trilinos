#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_Types.hpp"

namespace
{
  using namespace Intrepid2;

  using LineBasis = Basis_HGRAD_LINE_Cn_FEM<typename Kokkos::DefaultExecutionSpace::device_type,double,double>;
  using BasisBase = typename LineBasis::BasisBase;
  using BasisPtr = Teuchos::RCP<BasisBase>;
  using OutputViewType = typename BasisBase::OutputViewType;
  using ExecutionSpace  = typename BasisBase::ExecutionSpace;
  using OutputValueType = double;
  using PointValueType = double;

template<typename BasisBaseClass = void>
  class Basis_TensorBasis2
  :
  public BasisBaseClass
  {
  public:
    using BasisBase = BasisBaseClass;
    using BasisPtr  = Teuchos::RCP<BasisBase>;
  
  protected:
    BasisPtr basis1_;
    BasisPtr basis2_;
    
    std::vector<BasisPtr> tensorComponents_;
    
    std::string name_; // name of the basis
    
    int numTensorialExtrusions_; // relative to cell topo returned by getBaseCellTopology().
  public:
    using DeviceType = typename BasisBase::DeviceType;
    using ExecutionSpace  = typename BasisBase::ExecutionSpace;
    using OutputValueType = typename BasisBase::OutputValueType;
    using PointValueType  = typename BasisBase::PointValueType;
    
    using OrdinalTypeArray1DHost = typename BasisBase::OrdinalTypeArray1DHost;
    using OrdinalTypeArray2DHost = typename BasisBase::OrdinalTypeArray2DHost;
    using OutputViewType         = typename BasisBase::OutputViewType;
    using PointViewType          = typename BasisBase::PointViewType;
    using TensorBasis            = Basis_TensorBasis2<BasisBaseClass>;
  public:
    /** \brief  Constructor.
        \param [in] basis1 - the first component basis
        \param [in] basis2 - the second component basis
        \param [in] functionSpace - the function space to which the composite basis belongs (use FUNCTION_SPACE_MAX for unknown/unspecified function space)
        \param [in] useShardsCellTopologyAndTags - if true, attempt to assign a shards CellTopology corresponding to the tensor topology (shards Quad and Hex do not have tensor structure; this will map dofs appropriately) -- supported for 2D and 3D hypercubes
     */
    Basis_TensorBasis2(BasisPtr basis1, BasisPtr basis2, EFunctionSpace functionSpace = FUNCTION_SPACE_MAX,
                      const bool useShardsCellTopologyAndTags = false)
    :
    basis1_(basis1),basis2_(basis2)
    {
      this->functionSpace_ = functionSpace;
      
      Basis_TensorBasis2* basis1AsTensor = dynamic_cast<Basis_TensorBasis2*>(basis1_.get());
      if (basis1AsTensor)
      {
        auto basis1Components = basis1AsTensor->getTensorBasisComponents();
        tensorComponents_.insert(tensorComponents_.end(), basis1Components.begin(), basis1Components.end());
      }
      else
      {
        tensorComponents_.push_back(basis1_);
      }
      
      Basis_TensorBasis2* basis2AsTensor = dynamic_cast<Basis_TensorBasis2*>(basis2_.get());
      if (basis2AsTensor)
      {
        auto basis2Components = basis2AsTensor->getTensorBasisComponents();
        tensorComponents_.insert(tensorComponents_.end(), basis2Components.begin(), basis2Components.end());
      }
      else
      {
        tensorComponents_.push_back(basis2_);
      }
      
      this->basisCardinality_  = basis1->getCardinality() * basis2->getCardinality();
      this->basisDegree_       = std::max(basis1->getDegree(), basis2->getDegree());
      
      {
        std::ostringstream basisName;
        basisName << basis1->getName() << " x " << basis2->getName();
        name_ = basisName.str();
      }
      
      // set cell topology
      this->basisCellTopology_ = tensorComponents_[0]->getBaseCellTopology();
      this->numTensorialExtrusions_ = tensorComponents_.size() - 1;
      
      this->basisType_         = basis1_->getBasisType();
      this->basisCoordinates_  = COORDINATES_CARTESIAN;
      
      ordinal_type spaceDim1 = basis1_->getDomainDimension();
      ordinal_type spaceDim2 = basis2_->getDomainDimension();
      
      INTREPID2_TEST_FOR_EXCEPTION(spaceDim2 != 1, std::invalid_argument, "TensorBasis only supports 1D bases in basis2_ position");
      
      if (this->getBasisType() == BASIS_FEM_HIERARCHICAL)
      {
        // fill in degree lookup:
        int degreeSize = basis1_->getPolynomialDegreeLength() + basis2_->getPolynomialDegreeLength();
        this->fieldOrdinalPolynomialDegree_   = OrdinalTypeArray2DHost("TensorBasis - field ordinal polynomial degree", this->basisCardinality_, degreeSize);
        this->fieldOrdinalH1PolynomialDegree_ = OrdinalTypeArray2DHost("TensorBasis - field ordinal polynomial H^1 degree", this->basisCardinality_, degreeSize);
        
        const ordinal_type basis1Cardinality = basis1_->getCardinality();
        const ordinal_type basis2Cardinality = basis2_->getCardinality();
        
        int degreeLengthField1 = basis1_->getPolynomialDegreeLength();
        int degreeLengthField2 = basis2_->getPolynomialDegreeLength();
        
        for (ordinal_type fieldOrdinal1 = 0; fieldOrdinal1 < basis1Cardinality; fieldOrdinal1++)
        {
          OrdinalTypeArray1DHost degreesField1   = basis1_->getPolynomialDegreeOfField(fieldOrdinal1);
          OrdinalTypeArray1DHost h1DegreesField1 = basis1_->getH1PolynomialDegreeOfField(fieldOrdinal1);
          for (ordinal_type fieldOrdinal2 = 0; fieldOrdinal2 < basis2Cardinality; fieldOrdinal2++)
          {
            OrdinalTypeArray1DHost degreesField2   = basis2_->getPolynomialDegreeOfField(fieldOrdinal2);
            OrdinalTypeArray1DHost h1DegreesField2 = basis2_->getH1PolynomialDegreeOfField(fieldOrdinal2);
            const ordinal_type tensorFieldOrdinal = fieldOrdinal2 * basis1Cardinality + fieldOrdinal1;
            
            for (int d3=0; d3<degreeLengthField1; d3++)
            {
              this->fieldOrdinalPolynomialDegree_  (tensorFieldOrdinal,d3) =   degreesField1(d3);
              this->fieldOrdinalH1PolynomialDegree_(tensorFieldOrdinal,d3) = h1DegreesField1(d3);
            }
            for (int d3=0; d3<degreeLengthField2; d3++)
            {
              this->fieldOrdinalPolynomialDegree_  (tensorFieldOrdinal,d3+degreeLengthField1) =   degreesField2(d3);
              this->fieldOrdinalH1PolynomialDegree_(tensorFieldOrdinal,d3+degreeLengthField1) = h1DegreesField2(d3);
            }
          }
        }
      }
      
      if (useShardsCellTopologyAndTags)
      {
        setShardsTopologyAndTags();
      }
      else
      {
        // we build tags recursively, making reference to basis1_ and basis2_'s tags to produce the tensor product tags.
  //      // initialize tags
        const auto & cardinality = this->basisCardinality_;
  
        // Basis-dependent initializations
        const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
        const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
        const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
        const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
        const ordinal_type posDfCnt = 3;        // position in the tag, counting from 0, of DoF count for the subcell
  
        OrdinalTypeArray1DHost tagView("tag view", cardinality*tagSize);
  
        // we assume that basis2_ is defined on a line, and that basis1_ is defined on a domain that is once-extruded in by that line.
        auto cellTopo = CellTopology::cellTopology(this->basisCellTopology_, numTensorialExtrusions_);
        auto basis1Topo = cellTopo->getTensorialComponent();
        
        const ordinal_type spaceDim = spaceDim1 + spaceDim2;
        const ordinal_type sideDim   = spaceDim - 1;
        
        const OrdinalTypeArray2DHost ordinalToTag1 = basis1_->getAllDofTags();
        const OrdinalTypeArray2DHost ordinalToTag2 = basis2_->getAllDofTags();
                
        for (int fieldOrdinal1=0; fieldOrdinal1<basis1_->getCardinality(); fieldOrdinal1++)
        {
          ordinal_type subcellDim1   = ordinalToTag1(fieldOrdinal1,posScDim);
          ordinal_type subcellOrd1   = ordinalToTag1(fieldOrdinal1,posScOrd);
          ordinal_type subcellDfCnt1 = ordinalToTag1(fieldOrdinal1,posDfCnt);
          for (int fieldOrdinal2=0; fieldOrdinal2<basis2_->getCardinality(); fieldOrdinal2++)
          {
            ordinal_type subcellDim2   = ordinalToTag2(fieldOrdinal2,posScDim);
            ordinal_type subcellOrd2   = ordinalToTag2(fieldOrdinal2,posScOrd);
            ordinal_type subcellDfCnt2 = ordinalToTag2(fieldOrdinal2,posDfCnt);
            
            ordinal_type subcellDim = subcellDim1 + subcellDim2;
            ordinal_type subcellOrd;
            if (subcellDim2 == 0)
            {
              // vertex node in extrusion; the subcell is not extruded but belongs to one of the two "copies"
              // of the basis1 topology
              ordinal_type sideOrdinal = cellTopo->getTensorialComponentSideOrdinal(subcellOrd2); // subcellOrd2 is a "side" of the line topology
              subcellOrd = CellTopology::getSubcellOrdinalMap(cellTopo, sideDim, sideOrdinal,
                                                              subcellDim1, subcellOrd1);
            }
            else
            {
              // line subcell in time; the subcell *is* extruded in final dimension
              subcellOrd = cellTopo->getExtrudedSubcellOrdinal(subcellDim1, subcellOrd1);
              if (subcellOrd == -1)
              {
                std::cout << "ERROR: -1 subcell ordinal.\n";
                subcellOrd = cellTopo->getExtrudedSubcellOrdinal(subcellDim1, subcellOrd1);
              }
            }
            ordinal_type tensorFieldOrdinal = fieldOrdinal2 * basis1_->getCardinality() + fieldOrdinal1;
      //        cout << "(" << fieldOrdinal1 << "," << fieldOrdinal2 << ") --> " << i << endl;
            ordinal_type dofOffsetOrdinal1 = ordinalToTag1(fieldOrdinal1,posDfOrd);
            ordinal_type dofOffsetOrdinal2 = ordinalToTag2(fieldOrdinal2,posDfOrd);
            ordinal_type dofsForSubcell1   = ordinalToTag1(fieldOrdinal1,posDfCnt);
            ordinal_type dofOffsetOrdinal  = dofOffsetOrdinal2 * dofsForSubcell1 + dofOffsetOrdinal1;
            tagView(tagSize*tensorFieldOrdinal + posScDim) = subcellDim; // subcellDim
            tagView(tagSize*tensorFieldOrdinal + posScOrd) = subcellOrd; // subcell ordinal
            tagView(tagSize*tensorFieldOrdinal + posDfOrd) = dofOffsetOrdinal;  // ordinal of the specified DoF relative to the subcell
            tagView(tagSize*tensorFieldOrdinal + posDfCnt) = subcellDfCnt1 * subcellDfCnt2; // total number of DoFs associated with the subcell
          }
        }
        
        //        // Basis-independent function sets tag and enum data in tagToOrdinal_ and ordinalToTag_ arrays:
        //        // tags are constructed on host
        this->setOrdinalTagData(this->tagToOrdinal_,
                                this->ordinalToTag_,
                                tagView,
                                this->basisCardinality_,
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);
      }
    }
    
    void setShardsTopologyAndTags()
    {
// NOTE: this method matters (commenting it out gets us past the compiler failure)
      shards::CellTopology cellTopo1 = basis1_->getBaseCellTopology();
      shards::CellTopology cellTopo2 = basis2_->getBaseCellTopology();
      
      auto cellKey1 = basis1_->getBaseCellTopology().getKey();
      auto cellKey2 = basis2_->getBaseCellTopology().getKey();
      
      const int numTensorialExtrusions = basis1_->getNumTensorialExtrusions() + basis2_->getNumTensorialExtrusions();
      if ((cellKey1 == shards::Line<2>::key) && (cellKey2 == shards::Line<2>::key) && (numTensorialExtrusions == 0))
      {
        this->basisCellTopology_ = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
      }
      else if (   ((cellKey1 == shards::Quadrilateral<4>::key) && (cellKey2 == shards::Line<2>::key))
               || ((cellKey2 == shards::Quadrilateral<4>::key) && (cellKey1 == shards::Line<2>::key))
               || ((cellKey1 == shards::Line<2>::key) && (cellKey2 == shards::Line<2>::key) && (numTensorialExtrusions == 1))
              )
      {
        this->basisCellTopology_ = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() );
      }
      else
      {
        INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Cell topology combination not yet supported");
      }
      
      // numTensorialExtrusions_ is relative to the basisCellTopology_; what we've just done is found a cell topology of the same spatial dimension as the extruded topology, so now numTensorialExtrusions_ should be 0.
      numTensorialExtrusions_ = 0;
      
      // initialize tags
      {
        const auto & cardinality = this->basisCardinality_;
        
        // Basis-dependent initializations
        const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
        const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
        const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
        const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
        
        OrdinalTypeArray1DHost tagView("tag view", cardinality*tagSize);
        
        shards::CellTopology cellTopo = this->basisCellTopology_;
        
        ordinal_type tensorSpaceDim  = cellTopo.getDimension();
        ordinal_type spaceDim1       = cellTopo1.getDimension();
        ordinal_type spaceDim2       = cellTopo2.getDimension();
        
        TensorTopologyMap topoMap(cellTopo1, cellTopo2);
        
        for (ordinal_type d=0; d<=tensorSpaceDim; d++) // d: tensorial dimension
        {
          ordinal_type d2_max = std::min(spaceDim2,d);
          int subcellOffset = 0; // for this dimension of tensor subcells, how many subcells have we already counted with other d2/d1 combos?
          for (ordinal_type d2=0; d2<=d2_max; d2++)
          {
            ordinal_type d1 = d-d2;
            if (d1 > spaceDim1) continue;
            
            ordinal_type subcellCount2 = cellTopo2.getSubcellCount(d2);
            ordinal_type subcellCount1 = cellTopo1.getSubcellCount(d1);
            for (ordinal_type subcellOrdinal2=0; subcellOrdinal2<subcellCount2; subcellOrdinal2++)
            {
              ordinal_type subcellDofCount2 = basis2_->getDofCount(d2, subcellOrdinal2);
              for (ordinal_type subcellOrdinal1=0; subcellOrdinal1<subcellCount1; subcellOrdinal1++)
              {
                ordinal_type subcellDofCount1 = basis1_->getDofCount(d1, subcellOrdinal1);
                ordinal_type tensorLocalDofCount = subcellDofCount1 * subcellDofCount2;
                for (ordinal_type localDofID2 = 0; localDofID2<subcellDofCount2; localDofID2++)
                {
                  ordinal_type fieldOrdinal2 = basis2_->getDofOrdinal(d2, subcellOrdinal2, localDofID2);
                  OrdinalTypeArray1DHost degreesField2;
                  if (this->basisType_ == BASIS_FEM_HIERARCHICAL) degreesField2 = basis2_->getPolynomialDegreeOfField(fieldOrdinal2);
                  for (ordinal_type localDofID1 = 0; localDofID1<subcellDofCount1; localDofID1++)
                  {
                    ordinal_type fieldOrdinal1 = basis1_->getDofOrdinal(d1, subcellOrdinal1, localDofID1);
                    ordinal_type tensorLocalDofID = localDofID2 * subcellDofCount1 + localDofID1;
                    ordinal_type tensorFieldOrdinal = fieldOrdinal2 * basis1_->getCardinality() + fieldOrdinal1;
                    tagView(tensorFieldOrdinal*tagSize+0) = d; // subcell dimension
                    tagView(tensorFieldOrdinal*tagSize+1) = topoMap.getCompositeSubcellOrdinal(d1, subcellOrdinal1, d2, subcellOrdinal2);
                    tagView(tensorFieldOrdinal*tagSize+2) = tensorLocalDofID;
                    tagView(tensorFieldOrdinal*tagSize+3) = tensorLocalDofCount;
                  } // localDofID1
                } // localDofID2
              } // subcellOrdinal1
            } // subcellOrdinal2
            subcellOffset += subcellCount1 * subcellCount2;
          }
        }
        
        //        // Basis-independent function sets tag and enum data in tagToOrdinal_ and ordinalToTag_ arrays:
        //        // tags are constructed on host
        this->setOrdinalTagData(this->tagToOrdinal_,
                                this->ordinalToTag_,
                                tagView,
                                this->basisCardinality_,
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);
      }
    }
    
    virtual int getNumTensorialExtrusions() const override
    {
      return numTensorialExtrusions_;
    }
    
    ordinal_type getTensorDkEnumeration(ordinal_type dkEnum1, ordinal_type operatorOrder1,
                                        ordinal_type dkEnum2, ordinal_type operatorOrder2) const
    { }
    
    /** \brief Returns a simple decomposition of the specified operator: what operator(s) should be applied to basis1, and what operator(s) to basis2.  A one-element OperatorTensorDecomposition corresponds to a single TensorData entry; a multiple-element OperatorTensorDecomposition corresponds to a VectorData object with axialComponents = false.
     
     Subclasses must override this method.
    */
    virtual OperatorTensorDecomposition getSimpleOperatorDecomposition(const EOperator operatorType) const
    { }
    
    /** \brief Returns a full decomposition of the specified operator.  (Full meaning that all TensorBasis components are expanded into their non-TensorBasis components.)
      */
    virtual OperatorTensorDecomposition getOperatorDecomposition(const EOperator operatorType) const
    {
      if (((operatorType >= OPERATOR_D1) && (operatorType <= OPERATOR_D10)) || (operatorType == OPERATOR_GRAD))
      {
        // ordering of the operators is reverse-lexicographic, reading left to right (highest-dimension is fastest-moving).
        // first entry will be (operatorType, VALUE, …, VALUE)
        // next will be (operatorType - 1, OP_D1, VALUE, …, VALUE)
        // then         (operatorType - 1, VALUE, OP_D1, …, VALUE)
        
        ordinal_type numBasisComponents = tensorComponents_.size();
        
        auto opOrder = getOperatorOrder(operatorType); // number of derivatives that we take in total
        const int dkCardinality = getDkCardinality(operatorType, numBasisComponents);
        
        std::vector< std::vector<EOperator> > ops(dkCardinality);
        
        std::vector<EOperator> prevEntry(numBasisComponents, OPERATOR_VALUE);
        prevEntry[0] = operatorType;
        
        ops[0] = prevEntry;
        
        for (ordinal_type dkOrdinal=1; dkOrdinal<dkCardinality; dkOrdinal++)
        {
          std::vector<EOperator> entry = prevEntry;
          
          // decrement to follow reverse lexicographic ordering:
          ordinal_type cumulativeOpOrder = 0;
          ordinal_type finalOpOrder = getOperatorOrder(entry[numBasisComponents-1]);
          for (ordinal_type compOrdinal=0; compOrdinal<numBasisComponents; compOrdinal++)
          {
            const ordinal_type thisOpOrder = getOperatorOrder(entry[compOrdinal]);
            cumulativeOpOrder += thisOpOrder;
            if (cumulativeOpOrder + finalOpOrder == opOrder)
            {
              // decrement this
              EOperator decrementedOp;
              if (thisOpOrder == 1)
              {
                decrementedOp = OPERATOR_VALUE;
              }
              else
              {
                decrementedOp = static_cast<EOperator>(OPERATOR_D1 + ((thisOpOrder - 1) - 1));
              }
              entry[compOrdinal]   = decrementedOp;
              const ordinal_type remainingOpOrder = opOrder - cumulativeOpOrder + 1;
              entry[compOrdinal+1] = static_cast<EOperator>(OPERATOR_D1 + (remainingOpOrder - 1));
              for (ordinal_type i=compOrdinal+2; i<numBasisComponents; i++)
              {
                entry[i] = OPERATOR_VALUE;
              }
              break;
            }
          }
          ops[dkOrdinal] = entry;
          prevEntry = entry;
        }
        std::vector<double> weights(dkCardinality, 1.0);
        
        return OperatorTensorDecomposition(ops, weights);
      }
      else
      {
        OperatorTensorDecomposition opSimpleDecomposition = this->getSimpleOperatorDecomposition(operatorType);
        std::vector<BasisPtr> componentBases {basis1_, basis2_};
        return opSimpleDecomposition.expandedDecomposition(componentBases);
      }
    }
    
    /** \brief Allocate BasisValues container suitable for passing to the getValues() variant that takes a TensorPoints container as argument.
     
        The basic exact-sequence operators are supported (VALUE, GRAD, DIV, CURL), as are the Dn operators (OPERATOR_D1 through OPERATOR_D10).
     */
    virtual BasisValues<OutputValueType,DeviceType> allocateBasisValues( TensorPoints<PointValueType,DeviceType> points, const EOperator operatorType = OPERATOR_VALUE) const override
    {
     /* const bool operatorIsDk = (operatorType >= OPERATOR_D1) && (operatorType <= OPERATOR_D10);
      const bool operatorSupported = (operatorType == OPERATOR_VALUE) || (operatorType == OPERATOR_GRAD) || (operatorType == OPERATOR_CURL) || (operatorType == OPERATOR_DIV) || operatorIsDk;
      INTREPID2_TEST_FOR_EXCEPTION(!operatorSupported, std::invalid_argument, "operator is not supported by allocateBasisValues");
      
      // check that points's spatial dimension matches the basis
      const int spaceDim = this->getDomainDimension();
      INTREPID2_TEST_FOR_EXCEPTION(spaceDim != points.extent_int(1), std::invalid_argument, "points must be shape (P,D), with D equal to the dimension of the basis domain");
      
      // check that points has enough tensor components
      ordinal_type numBasisComponents = tensorComponents_.size();
      if (numBasisComponents > points.numTensorComponents())
      {
        // Then we require points to have a trivial tensor structure.  (Subclasses could be more sophisticated.)
        // (More sophisticated approaches are possible here, too, but likely the most common use case in which there is not a one-to-one correspondence
        //  between basis components and point components will involve trivial tensor structure in the points...)
        INTREPID2_TEST_FOR_EXCEPTION(points.numTensorComponents() != 1, std::invalid_argument, "If points does not have the same number of tensor components as the basis, then it should have trivial tensor structure.");
        const ordinal_type numPoints = points.extent_int(0);
        auto outputView = this->allocateOutputView(numPoints, operatorType);
        
        Data<OutputValueType,DeviceType> outputData(outputView);
        TensorData<OutputValueType,DeviceType> outputTensorData(outputData);
        
        return BasisValues<OutputValueType,DeviceType>(outputTensorData);
      }
      INTREPID2_TEST_FOR_EXCEPTION(numBasisComponents > points.numTensorComponents(), std::invalid_argument, "points must have at least as many tensorial components as basis.");
      
      OperatorTensorDecomposition opDecomposition = getOperatorDecomposition(operatorType);
            
      ordinal_type numVectorComponents = opDecomposition.numVectorComponents();
      const bool useVectorData = numVectorComponents > 1;
      
      std::vector<ordinal_type> componentPointCounts(numBasisComponents);
      ordinal_type pointComponentNumber = 0;
      for (ordinal_type r=0; r<numBasisComponents; r++)
      {
        const ordinal_type compSpaceDim = tensorComponents_[r]->getDomainDimension();
        ordinal_type dimsSoFar = 0;
        ordinal_type numPointsForBasisComponent = 1;
        while (dimsSoFar < compSpaceDim)
        {
          INTREPID2_TEST_FOR_EXCEPTION(pointComponentNumber >= points.numTensorComponents(), std::invalid_argument, "Error in processing points container; perhaps it is mis-sized?");
          const int numComponentPoints = points.componentPointCount(pointComponentNumber);
          const int numComponentDims = points.getTensorComponent(pointComponentNumber).extent_int(1);
          numPointsForBasisComponent *= numComponentPoints;
          dimsSoFar += numComponentDims;
          INTREPID2_TEST_FOR_EXCEPTION(dimsSoFar > points.numTensorComponents(), std::invalid_argument, "Error in processing points container; perhaps it is mis-sized?");
          pointComponentNumber++;
        }
        componentPointCounts[r] = numPointsForBasisComponent;
      }
      
      if (useVectorData)
      {
        const int numFamilies = 1;
        std::vector< std::vector<TensorData<OutputValueType,DeviceType> > > vectorComponents(numFamilies, std::vector<TensorData<OutputValueType,DeviceType> >(numVectorComponents));
        
        const int familyOrdinal = 0;
        for (ordinal_type vectorComponentOrdinal=0; vectorComponentOrdinal<numVectorComponents; vectorComponentOrdinal++)
        {
          if (!opDecomposition.identicallyZeroComponent(vectorComponentOrdinal))
          {
            std::vector< Data<OutputValueType,DeviceType> > componentData;
            for (ordinal_type r=0; r<numBasisComponents; r++)
            {
              const int numComponentPoints = componentPointCounts[r];
              const EOperator op = opDecomposition.op(vectorComponentOrdinal, r);
              auto componentView = tensorComponents_[r]->allocateOutputView(numComponentPoints, op);
              componentData.push_back(Data<OutputValueType,DeviceType>(componentView));
            }
            vectorComponents[familyOrdinal][vectorComponentOrdinal] = TensorData<OutputValueType,DeviceType>(componentData);
          }
        }
        VectorData<OutputValueType,DeviceType> vectorData(vectorComponents);
        return BasisValues<OutputValueType,DeviceType>(vectorData);
      }
      else
      {
        // TensorData: single tensor product
        std::vector< Data<OutputValueType,DeviceType> > componentData;
        
        const ordinal_type vectorComponentOrdinal = 0;
        for (ordinal_type r=0; r<numBasisComponents; r++)
        {
          const int numComponentPoints = componentPointCounts[r];
          const EOperator op = opDecomposition.op(vectorComponentOrdinal, r);
          auto componentView = tensorComponents_[r]->allocateOutputView(numComponentPoints, op);
          
          const int rank = 2; // (F,P) -- TensorData-only BasisValues are always scalar-valued.  Use VectorData for vector-valued.
          // (we need to be explicit about the rank argument because GRAD, even in 1D, elevates to rank 3), so e.g. DIV of HDIV uses a componentView that is rank 3;
          //  we want Data to insulate us from that fact)
          const Kokkos::Array<int,7> extents {componentView.extent_int(0), componentView.extent_int(1), 1,1,1,1,1};
          Kokkos::Array<DataVariationType,7> variationType {GENERAL, GENERAL, CONSTANT, CONSTANT, CONSTANT, CONSTANT, CONSTANT };
          componentData.push_back(Data<OutputValueType,DeviceType>(componentView, rank, extents, variationType));
        }
        
        TensorData<OutputValueType,DeviceType> tensorData(componentData);
        
        std::vector< TensorData<OutputValueType,DeviceType> > tensorDataEntries {tensorData};
        return BasisValues<OutputValueType,DeviceType>(tensorDataEntries);
      }
*/
    }
    
    // since the getValues() below only overrides the FEM variant, we specify that
    // we use the base class's getValues(), which implements the FVD variant by throwing an exception.
    // (It's an error to use the FVD variant on this basis.)
    using BasisBase::getValues;
    
    /** \brief  Method to extract component points from composite points.
        \param [in]  inputPoints                  - points defined on the composite cell topology
        \param [in]  attemptTensorDecomposition   - if true, attempt to find a tensor decomposition.
        \param [out] inputPoints1                 - points defined on the first component cell topology
        \param [out] inputPoints2                 - points defined on the second component cell topology
        \param [out] tensorDecompositionSucceeded - if true, the attempt to find a tensor decomposition succeeded.
     
     At present, attemptTensorDecomposition is ignored, and tensorDecompositionSucceeded will always return false.
     However, we intend to support the tensor decomposition in the future, which will allow substantial optimizations
     in computation of tensor bases.
     */
    void getComponentPoints(const PointViewType inputPoints, const bool attemptTensorDecomposition,
                            PointViewType & inputPoints1, PointViewType & inputPoints2, bool &tensorDecompositionSucceeded) const
    {
/*
      INTREPID2_TEST_FOR_EXCEPTION(attemptTensorDecomposition, std::invalid_argument, "tensor decomposition not yet supported");
      
      // for inputPoints that are actually tensor-product of component quadrature points (say),
      // having just the one input (which will have a lot of redundant point data) is suboptimal
      // The general case can have unique x/y/z coordinates at every point, though, so we have to support that
      // when this interface is used.  But we may try detecting that the data is tensor-product and compressing
      // from there...  Ultimately, we should also add a getValues() variant that takes multiple input point containers,
      // one for each tensorial dimension.
      
      // this initial implementation is intended to simplify development of 2D and 3D bases, while also opening
      // the possibility of higher-dimensional bases.  It is not necessarily optimized for speed/memory.  There
      // are things we can do in this regard, which may become important for matrix-free computations wherein
      // basis values don't get stored but are computed dynamically.
      
      int spaceDim1 = basis1_->getDomainDimension();
      int spaceDim2 = basis2_->getDomainDimension();
      
      int totalSpaceDim   = inputPoints.extent_int(1);
      
      TEUCHOS_ASSERT(spaceDim1 + spaceDim2 == totalSpaceDim);
      
      // first pass: just take subviews to get input points -- this will result in redundant computations when points are themselves tensor product (i.e., inputPoints itself contains redundant data)
      
      inputPoints1 = Kokkos::subview(inputPoints,Kokkos::ALL(),std::make_pair(0,spaceDim1));
      inputPoints2 = Kokkos::subview(inputPoints,Kokkos::ALL(),std::make_pair(spaceDim1,totalSpaceDim));
      
      //      std::cout << "inputPoints : " << inputPoints.extent(0) << " x " << inputPoints.extent(1) << std::endl;
      //      std::cout << "inputPoints1 : " << inputPoints1.extent(0) << " x " << inputPoints1.extent(1) << std::endl;
      //      std::cout << "inputPoints2 : " << inputPoints2.extent(0) << " x " << inputPoints2.extent(1) << std::endl;
      
      tensorDecompositionSucceeded = false;
*/
    }
    
    virtual void getDofCoords( typename BasisBase::ScalarViewType dofCoords ) const override
    {
      int spaceDim1 = basis1_->getBaseCellTopology().getDimension();
      int spaceDim2 = basis2_->getBaseCellTopology().getDimension();
      
      using ValueType    = typename BasisBase::ScalarViewType::value_type;
      using ResultLayout = typename DeduceLayout< typename BasisBase::ScalarViewType >::result_layout;
      using ViewType     = Kokkos::DynRankView<ValueType, ResultLayout, DeviceType >;
      
      const ordinal_type basisCardinality1 = basis1_->getCardinality();
      const ordinal_type basisCardinality2 = basis2_->getCardinality();

      ViewType dofCoords1("dofCoords1",basisCardinality1,spaceDim1);
      ViewType dofCoords2("dofCoords2",basisCardinality2,spaceDim2);
      
      basis1_->getDofCoords(dofCoords1);
      basis2_->getDofCoords(dofCoords2);
      
      Kokkos::RangePolicy<ExecutionSpace> policy(0, basisCardinality2);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const int fieldOrdinal2)
       {
         for (int fieldOrdinal1=0; fieldOrdinal1<basisCardinality1; fieldOrdinal1++)
         {
           const ordinal_type fieldOrdinal = fieldOrdinal1 + fieldOrdinal2 * basisCardinality1;
           for (int d1=0; d1<spaceDim1; d1++)
           {
             dofCoords(fieldOrdinal,d1) = dofCoords1(fieldOrdinal1,d1);
           }
           for (int d2=0; d2<spaceDim2; d2++)
           {
             dofCoords(fieldOrdinal,spaceDim1+d2) = dofCoords2(fieldOrdinal2,d2);
           }
         }
       });
    }
    

    /** \brief  Fills in coefficients of degrees of freedom on the reference cell
        \param [out] dofCoeffs - the container into which to place the degrees of freedom.

     dofCoeffs is a rank 1 with dimension equal to the cardinality of the basis.

     Note that getDofCoeffs() is not supported by all bases; in particular, hierarchical bases do not generally support this.
     */
    virtual void getDofCoeffs( typename BasisBase::ScalarViewType dofCoeffs ) const override
    {
      using ValueType    = typename BasisBase::ScalarViewType::value_type;
      using ResultLayout = typename DeduceLayout< typename BasisBase::ScalarViewType >::result_layout;
      using ViewType     = Kokkos::DynRankView<ValueType, ResultLayout, DeviceType >;

      ViewType dofCoeffs1("dofCoeffs1",basis1_->getCardinality());
      ViewType dofCoeffs2("dofCoeffs2",basis2_->getCardinality());

      basis1_->getDofCoeffs(dofCoeffs1);
      basis2_->getDofCoeffs(dofCoeffs2);

      const ordinal_type basisCardinality1 = basis1_->getCardinality();
      const ordinal_type basisCardinality2 = basis2_->getCardinality();

      Kokkos::RangePolicy<ExecutionSpace> policy(0, basisCardinality2);
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const int fieldOrdinal2)
       {
         for (int fieldOrdinal1=0; fieldOrdinal1<basisCardinality1; fieldOrdinal1++)
         {
           const ordinal_type fieldOrdinal = fieldOrdinal1 + fieldOrdinal2 * basisCardinality1;
           dofCoeffs(fieldOrdinal) = dofCoeffs1(fieldOrdinal1);
           dofCoeffs(fieldOrdinal) = dofCoeffs2(fieldOrdinal2);
         }
       });
    }

    /** \brief  Returns basis name
     
     \return the name of the basis
     */
    virtual
    const char*
    getName() const override {
      return name_.c_str();
    }
    
    std::vector<BasisPtr> getTensorBasisComponents() const
    {
      return tensorComponents_;
    }
    
    virtual
    void
    getValues(       BasisValues<OutputValueType,DeviceType> outputValues,
               const TensorPoints<PointValueType,DeviceType>  inputPoints,
               const EOperator operatorType = OPERATOR_VALUE ) const override
    { }
    
    void getValues( OutputViewType outputValues, const PointViewType  inputPoints,
                   const EOperator operatorType = OPERATOR_VALUE ) const override
    {
      const bool tensorPoints = false;
     OutputViewType outputValues1, outputValues2;
               
    int basisCardinality1 = 1;
    const int vectorSize = 1;
    auto policy = Kokkos::TeamPolicy<ExecutionSpace>(basisCardinality1,Kokkos::AUTO(),vectorSize);
    
    double weight = 1.0;
    using FunctorType = TensorViewFunctor<ExecutionSpace, OutputValueType, OutputViewType>;
    auto outputValues1_dkEnum1 = Kokkos::subview(outputValues1,Kokkos::ALL(),Kokkos::ALL());
    auto outputValues2_dkEnum2 = Kokkos::subview(outputValues2,Kokkos::ALL(),Kokkos::ALL());
    ordinal_type dkTensorIndex = 1;
    auto outputValues_dkTensor = Kokkos::subview(outputValues,Kokkos::ALL(),Kokkos::ALL(),dkTensorIndex);
    FunctorType functor(outputValues_dkTensor, outputValues1_dkEnum1, outputValues2_dkEnum2, tensorPoints, weight);
    Kokkos::parallel_for( policy , functor, "TensorViewFunctor");
    
    }
    
    virtual void getValues(OutputViewType outputValues, const EOperator operatorType,
                           const PointViewType  inputPoints1, const PointViewType  inputPoints2,
                           bool tensorPoints) const
    {
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "one-operator, two-inputPoints getValues should be overridden by TensorBasis subclasses");
    }
    
    void getValues( OutputViewType outputValues,
                   const PointViewType  inputPoints1, const EOperator operatorType1,
                   const PointViewType  inputPoints2, const EOperator operatorType2,
                   bool tensorPoints, double weight=1.0) const
    { }
  }; // Basis_TensorBasis2

    // just calling the below succeeds…
    /*void getValuesFree( )
    {
      const bool tensorPoints = false;
     OutputViewType outputValues;
     OutputViewType outputValues1, outputValues2;
                
    int basisCardinality1 = 1;
    const int vectorSize = 1;
    auto policy = Kokkos::TeamPolicy<ExecutionSpace>(basisCardinality1,Kokkos::AUTO(),vectorSize);
    
    double weight = 1.0;
    using FunctorType = TensorViewFunctor<ExecutionSpace, OutputValueType, OutputViewType>;
    auto outputValues1_dkEnum1 = Kokkos::subview(outputValues1,Kokkos::ALL(),Kokkos::ALL());
    auto outputValues2_dkEnum2 = Kokkos::subview(outputValues2,Kokkos::ALL(),Kokkos::ALL()); 
    ordinal_type dkTensorIndex = 1;
    auto outputValues_dkTensor = Kokkos::subview(outputValues,Kokkos::ALL(),Kokkos::ALL(),dkTensorIndex);
    FunctorType functor(outputValues_dkTensor, outputValues1_dkEnum1, outputValues2_dkEnum2, tensorPoints, weight);
    Kokkos::parallel_for( policy , functor, "TensorViewFunctor"); 
      
    }*/
  TEUCHOS_UNIT_TEST( BasisCardinality, Hypercube )
  {
   const int polyDegree   = 1;
//   getValuesFree();
   BasisPtr lineBasis = Teuchos::rcp(new LineBasis(polyDegree) );
   BasisPtr tensorBasis = lineBasis;
   tensorBasis = Teuchos::rcp(new Basis_TensorBasis2<BasisBase>(tensorBasis, lineBasis, FUNCTION_SPACE_HGRAD));
   int expectedExtrusionCount = 1;
   TEST_EQUALITY(expectedExtrusionCount, tensorBasis->getNumTensorialExtrusions());
  }

} // namespace

