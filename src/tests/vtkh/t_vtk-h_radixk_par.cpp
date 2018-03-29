//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_dataset.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <mpi.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/rendering/Image.hpp>
#include <vtkh/rendering/compositing/RadixKCompositor.hpp>
#include "t_test_utils.hpp"

#include <iostream>



//----------------------------------------------------------------------------
TEST(vtkh_radixk, vtkh_parallel_composite)
{
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  vtkh::SetMPIComm(MPI_COMM_WORLD);
  
  vtkm::Bounds image_size;
  image_size.X.Min = 1;
  image_size.Y.Min = 1;
  image_size.X.Max = 512;
  image_size.Y.Max = 512;
  vtkh::Image image(image_size);
   
  diy::mpi::communicator diy_comm = diy::mpi::communicator(vtkh::GetMPIComm());
  vtkh::RadixKCompositor comp; 
  comp.CompositeSurface(diy_comm, image);

  MPI_Finalize();
}
