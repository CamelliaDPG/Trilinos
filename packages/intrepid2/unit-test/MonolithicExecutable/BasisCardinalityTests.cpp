#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_Types.hpp"

namespace
{
  using namespace Intrepid2;

  using LineBasis = Basis_HGRAD_LINE_Cn_FEM<typename Kokkos::DefaultExecutionSpace::device_type,double,double>;
  using BasisBase = typename LineBasis::BasisBase;
  using BasisPtr = Teuchos::RCP<BasisBase>;

  static BasisPtr getHypercubeBasis_HGRAD2(int polyOrder, int spaceDim)
  {
    using Teuchos::rcp;

    BasisPtr lineBasis = Teuchos::rcp(new LineBasis(polyOrder) );
    BasisPtr tensorBasis = lineBasis;

    for (int d=1; d<spaceDim; d++)
    {
      tensorBasis = Teuchos::rcp(new Basis_TensorBasis<BasisBase>(tensorBasis, lineBasis, FUNCTION_SPACE_HGRAD));
    }

    return tensorBasis;
  }

  TEUCHOS_UNIT_TEST( BasisCardinality, Hypercube )
  {
   const int spaceDim     = 1;
   const int polyDegree   = 1;
   auto nodalBasis        = getHypercubeBasis_HGRAD2(polyDegree, spaceDim);
   // the following succeeds:
   //auto nodalBasis = getLineBasis<NodalBasisFamily>(FUNCTION_SPACE_HGRAD, polyDegree);
   int expectedExtrusionCount = spaceDim - 1;
   TEST_EQUALITY(expectedExtrusionCount, nodalBasis->getNumTensorialExtrusions());
  }

} // namespace

