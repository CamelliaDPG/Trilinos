#include "Teuchos_GlobalMPISession.hpp"

#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_DefaultComm.hpp"

#include "Kokkos_Core.hpp"

#include "Intrepid2_Data.hpp"
#include "Intrepid2_TestUtils.hpp"
#include "Intrepid2_Types.hpp"

enum CaseChoice
{
  Constant,
  Affine,
  General
};

std::string to_string(CaseChoice choice)
{
  switch (choice) {
    case Constant: return "Constant";
    case Affine:   return "Affine";
    case General:  return "General";
    
    default:       return "Unknown CaseChoice";
  }
}

using namespace Intrepid2;

static const int NUM_CELLS = 16000;

template< typename Scalar, typename DeviceType >
inline
Data<Scalar, DeviceType> getData(CaseChoice caseChoice, const int numPoints, const double baseValue)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  const int numCells = NUM_CELLS;
  Kokkos::Array<ordinal_type,2> extents {numCells, numPoints};
  Kokkos::Array<DataVariationType,2> variationTypes {GENERAL,GENERAL};
  
  switch (caseChoice) {
    case Constant:
      return Data<Scalar, DeviceType>(baseValue,extents);
    case Affine:
    {
      // (C,P); varies in C dimension
      variationTypes[1] = CONSTANT;
      Kokkos::View<Scalar*,DeviceType> cellView("affine case - underlying view",numCells);
      Kokkos::RangePolicy<ExecutionSpace> policy(ExecutionSpace(), 0, numCells);
      Kokkos::parallel_for("initialize underlying view data", policy,
      KOKKOS_LAMBDA (const int &i0) {
        cellView(i0) = i0 * baseValue;
      });
      return Data<Scalar, DeviceType>(cellView,extents,variationTypes);
    }
    case General:
    {
      // (C,P); varies in C and P dimensions
      variationTypes[1] = GENERAL;
      Kokkos::View<Scalar**,DeviceType> cellView("affine case - underlying view",numCells,numPoints);
      Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{numCells,numPoints});
      Kokkos::parallel_for("initialize underlying view data", policy,
      KOKKOS_LAMBDA (const int &i0, const int &i1) {
        cellView(i0,i1) = i0 * baseValue + i1;
      });
      return Data<Scalar, DeviceType>(cellView,extents,variationTypes);
    }
    default:
      return Data<Scalar, DeviceType>();
  }
}

double idealSpeedup(CaseChoice caseChoice, const int numPoints)
{
  switch (caseChoice) {
    case Constant:
      return NUM_CELLS * numPoints;
    case Affine:
      return numPoints;
    case General:
      return 1.0;
    default:
      return -1.0;
  }
}

template< typename Scalar, typename DeviceType >
Kokkos::View<Scalar**, DeviceType> allocateView(const int numPoints)
{
  Kokkos::View<Scalar**,DeviceType> view("DataCombinationPerformance - View", NUM_CELLS, numPoints);
  return view;
}

template< typename Scalar, typename DeviceType >
inline
void fillView(CaseChoice caseChoice, Kokkos::View<Scalar**,DeviceType> view, const double baseValue)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  
  switch (caseChoice) {
    case Constant:
      Kokkos::deep_copy(view, baseValue);
      break;
    case Affine:
    {
      Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{view.extent_int(0),view.extent_int(1)});
      // (C,P); varies in C dimension
      Kokkos::parallel_for("initialize underlying view data", policy,
      KOKKOS_LAMBDA (const int &i0, const int &i1) {
        view(i0,i1) = i0 * baseValue;
      });
    }
      break;
    case General:
    {
      Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{view.extent_int(0),view.extent_int(1)});
      // (C,P); varies in C and P dimensions
      Kokkos::parallel_for("initialize underlying view data", policy,
      KOKKOS_LAMBDA (const int &i0, const int &i1) {
        view(i0,i1) = i0 * baseValue + i1;
      });
    }
    break;
    default:
      break;
  }
  ExecutionSpace().fence();
}

template< typename Scalar, typename DeviceType >
void sumViews(Kokkos::View<Scalar**,DeviceType> resultView,
              Kokkos::View<Scalar**,DeviceType> view1, Kokkos::View<Scalar**,DeviceType> view2)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{resultView.extent_int(0),resultView.extent_int(1)});
  
  Kokkos::parallel_for("initialize underlying view data", policy,
  KOKKOS_LAMBDA (const int &i0, const int &i1) {
    resultView(i0,i1) = view1(i0,i1) + view2(i0,i1);
  });
}

int main( int argc, char* argv[] )
{
  // Note that the dtor for GlobalMPISession will call Kokkos::finalize_all() but does not call Kokkos::initialize()...
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Kokkos::initialize(argc,argv);
  
  using std::cout;
  using std::endl;
  using std::string;
  using std::vector;
  
  bool success = true;
  
  {
    vector<CaseChoice> allCaseChoices {Constant, Affine, General};
    
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    string caseChoiceString = "All"; // alternatives: Standard, NonAffineTensor, AffineTensor, Uniform
    
    int pointCountFixed = -1;
    int pointCountMin = 16;
    int pointCountMax = 1024;
    
    cmdp.setOption("case", &caseChoiceString, "Options: All, Constant, Affine, General");
    cmdp.setOption("pointCount", &pointCountFixed, "Single point count to run with");
    cmdp.setOption("minPointCount", &pointCountMin, "Starting point count (will double until max count is reached)");
    cmdp.setOption("maxPointCount", &pointCountMax, "Maximum point count");
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }

    vector<CaseChoice> caseChoices;
    if (caseChoiceString == "All")
    {
      caseChoices = allCaseChoices;
    }
    else if (caseChoiceString == "Constant")
    {
      caseChoices = vector<CaseChoice>{Constant};
    }
    else if (caseChoiceString == "Affine")
    {
      caseChoices = vector<CaseChoice>{Affine};
    }
    else if (caseChoiceString == "General")
    {
      caseChoices = vector<CaseChoice>{General};
    }
    else
    {
      cout << "Unrecognized case choice: " << caseChoiceString << endl;
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    if (pointCountFixed > 0)
    {
      pointCountMin = pointCountFixed;
      pointCountMax = pointCountFixed;
    }
    
    using Scalar = double;
    using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
    
    using DataType = Data<Scalar, DeviceType>;
    
    const int charWidth = 15;
    using std::vector;
    using std::map;
    using std::pair;
    using std::make_pair;
    using std::tuple;
    using std::cout;
    using std::endl;
    using std::setw;
    using std::scientific;
    using std::fixed;
    
    const double absTol = 1e-15, relTol = 1e-15;
    
    for (CaseChoice caseChoice1 : caseChoices)
    {
      for (CaseChoice caseChoice2 : caseChoices)
      {
        // since constant takes so little time (and measurement is therefore noisy), we do a bunch of measurements and use their average
        const bool bothConstant   = (caseChoice1 == Constant) && (caseChoice2 == Constant);
        const int numMeasurements = bothConstant ? 1000 : 1;
        
        cout << "\n\n*******************************************\n";
        cout <<     "******   " << setw(12) << to_string(caseChoice1) << "/" << to_string(caseChoice2) << setw(14) << "   ******\n";
        cout << "*******************************************\n";
        for (int pointCount=pointCountMin; pointCount<=pointCountMax; pointCount *= 2)
        {
          const double baseValue1 = M_PI;
          const double baseValue2 = 1.0;

          Data<Scalar, DeviceType> result;
          auto dataTimer = Teuchos::TimeMonitor::getNewTimer("Data sum");
          for (int i=0; i<numMeasurements; i++)
          {
            auto data1 = getData<Scalar, DeviceType>(caseChoice1, pointCount, baseValue1);
            auto data2 = getData<Scalar, DeviceType>(caseChoice2, pointCount, baseValue2);
            
            result = DataType::allocateInPlaceCombinationResult(data1, data2);
            
            DeviceType::execution_space().fence();
            dataTimer->start();
            result.storeInPlaceSum(data1, data2);
            DeviceType::execution_space().fence();
            dataTimer->stop();
          }
          double dataElapsedTimeSeconds = dataTimer->totalElapsedTime() / numMeasurements;
          
          cout << "Point count:          " << setw(charWidth) << pointCount << endl;
          cout << "Time (sum - data):    " << setw(charWidth) << std::setprecision(2) << scientific << dataElapsedTimeSeconds << endl;
          
          dataTimer->reset();
          
          auto viewTimer = Teuchos::TimeMonitor::getNewTimer("View sum");
          auto view1 = allocateView<Scalar, DeviceType>(pointCount);
          auto view2 = allocateView<Scalar, DeviceType>(pointCount);
          auto resultView = allocateView<Scalar, DeviceType>(pointCount);
          
          fillView(caseChoice1, view1, baseValue1);
          fillView(caseChoice2, view2, baseValue2);
          
          DeviceType::execution_space().fence();
          viewTimer->start();
          sumViews(resultView, view1, view2);
          DeviceType::execution_space().fence();
          viewTimer->stop();
          double viewElapsedTimeSeconds = viewTimer->totalElapsedTime();
          cout << "Time (sum - view):    " << setw(charWidth) << std::setprecision(2) << scientific << viewElapsedTimeSeconds << endl;
          
          viewTimer->reset();
          
          const double maxSpeedup = std::min(idealSpeedup(caseChoice1, pointCount),idealSpeedup(caseChoice2, pointCount));
          const double actualSpeedup = viewElapsedTimeSeconds / dataElapsedTimeSeconds;
          const double percentage = actualSpeedup / maxSpeedup * 100.0;
          cout << "Ideal speedup:        " << setw(charWidth) << std::setprecision(2) << scientific << maxSpeedup << endl;
          cout << "Actual speedup:       " << setw(charWidth) << std::setprecision(2) << scientific << actualSpeedup << endl;
          cout << "Percentage of ideal:  " << setw(charWidth) << std::setprecision(2) << fixed << percentage << "%" << endl;
          cout << endl;
          
          // to optimize for the case where the test passes, we output to a Teuchos::oblackholestream first.
          // if the test fails, we repeat the comparison to std::cout.
          Teuchos::oblackholestream  outNothing;
          Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&outNothing,false));
          bool localSuccess = true;
          testFloatingEquality2(resultView, result, relTol, absTol, out, localSuccess);
        }
      }
    }
  }
  
  if (success)
    return 0;
  else
    return -1;
}
