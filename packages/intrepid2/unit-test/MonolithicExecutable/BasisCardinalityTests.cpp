#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_Types.hpp"

namespace
{
  using namespace Intrepid2;

  TEUCHOS_UNIT_TEST( BasisCardinality, Hypercube )
  {
   using NodalBasisFamily = NodalBasisFamily<typename Kokkos::DefaultExecutionSpace::device_type>;
   const int spaceDim     = 1;
   const int polyDegree   = 1;
   auto nodalBasis        = getHypercubeBasis_HGRAD<NodalBasisFamily>(polyDegree, spaceDim);
   int expectedExtrusionCount = spaceDim - 1;
   TEST_EQUALITY(expectedExtrusionCount, nodalBasis->getNumTensorialExtrusions());
  }

} // namespace
