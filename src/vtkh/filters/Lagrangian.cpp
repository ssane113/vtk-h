#include <iostream>
#include <vtkh/vtkm_filters/vtkmLagrangian.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/filters/GhostStripper.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/VariantArrayHandle.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/ArrayPortalToIterators.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/ParticleAdvection.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/particleadvection/GridEvaluators.h>
#include <vtkm/worklet/particleadvection/Integrators.h>
#include <vtkm/worklet/particleadvection/Particles.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/cont/DataSetFieldAdd.h>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

static int cycle = 0;
static vtkm::Id maxID = 0;
static double pt[3];
static double* rank_bounds; 
static int seed_dims[3];
static std::vector<vtkm::Vec<vtkm::FloatDefault, 3>> next, initpts, ex_next;
static std::vector<vtkm::Id> seedValidity, seedID, seedRank, ex_valid, ex_ID, ex_orig, ex_rank;
static double BB[6];

namespace vtkh
{

Lagrangian::Lagrangian()
{

}

Lagrangian::~Lagrangian()
{

}

void
Lagrangian::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
Lagrangian::SetStepSize(const double &step_size)
{
  m_step_size = step_size;
}

void
Lagrangian::SetWriteFrequency(const int &write_frequency)
{
  m_write_frequency = write_frequency;
}

void
Lagrangian::SetCustomSeedResolution(const int &cust_res)
{
	m_cust_res = cust_res;
}

void
Lagrangian::SetSeedResolutionInX(const int &x_res)
{
	m_x_res = x_res;
}

void
Lagrangian::SetSeedResolutionInY(const int &y_res)
{
	m_y_res = y_res;
}
void
Lagrangian::SetSeedResolutionInZ(const int &z_res)
{
	m_z_res = z_res;
}


void Lagrangian::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void Lagrangian::PostExecute()
{
  Filter::PostExecute();
}

void Lagrangian::DoExecute()
{
#ifdef VTKH_PARALLEL
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();
  if(num_domains != 1)
  {
    throw Error("Lagrangian analysis can currently only operate on a single domain per MPI rank.");
  }
  else
  { // num_domain = 1
   
/**** Prepare and check input -- Strip ghost zones, check for vector field ****/ 
/*    vtkh::GhostStripper stripper;
    stripper.SetInput(this->m_input);
    stripper.SetField("ascent_ghosts");
    stripper.Update();
    vtkh::DataSet *min_dom = stripper.GetOutput();
*/
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(0, dom, domain_id);
//    min_dom->GetDomain(0, dom, domain_id); 
    
    if(dom.HasField(m_field_name))
    {   
      using vectorField_d = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>>;
      using vectorField_f = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3>>;
      auto field = dom.GetField(m_field_name).GetData();    
      if(!field.IsType<vectorField_d>() && !field.IsType<vectorField_f>())
      {   
        throw Error("Vector field type does not match <vtkm::Vec<vtkm::Float32,3>> or <vtkm::Vec<vtkm::Float64,3>>");
      }    
    }   
    else
    {   
      throw Error("Domain does not contain specified vector field for Lagrangian analysis.");
    }   
    

/**** MPI Information *****/

    vtkm::Id rank = vtkh::GetMPIRank();
    vtkm::Id num_ranks = vtkh::GetMPISize();
    bool allReceived[num_ranks] = {false};
    allReceived[rank] = true;


/**** Initialization process for every rank ****/

 
    if(cycle == 0)     
    {
/****  Setting up bounding box and neighborhood information to evaluate containing domain  *****/

      rank_bounds = (double*)malloc(sizeof(double) * num_ranks * 6);
      vtkm::Bounds b = dom.GetCoordinateSystem().GetBounds();
      rank_bounds[6*rank + 0] = b.X.Min;
      rank_bounds[6*rank + 1] = b.X.Max;
      rank_bounds[6*rank + 2] = b.Y.Min;
      rank_bounds[6*rank + 3] = b.Y.Max;
      rank_bounds[6*rank + 4] = b.Z.Min;
      rank_bounds[6*rank + 5] = b.Z.Max;

      BB[0] = b.X.Min;
      BB[1] = b.X.Max;
      BB[2] = b.Y.Min;
      BB[3] = b.Y.Max;
      BB[4] = b.Z.Min;
      BB[5] = b.Z.Max;
  	
      int bufsize_1 = (6 * num_ranks)*10 + (MPI_BSEND_OVERHEAD * num_ranks);
      double *buf_1 = (double*)malloc(sizeof(double)*bufsize_1);
      MPI_Buffer_attach(buf_1, bufsize_1);
      
      for(int i = 0; i < num_ranks; i++)
      {
        if(i != rank)
        {
          int ierr = MPI_Bsend(BB, 6, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
      }
        
      while(!AllMessagesReceived(allReceived, num_ranks))
      {
        MPI_Status probe_status, recv_status;
        int ierr = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &probe_status);
        int count;
        MPI_Get_count(&probe_status, MPI_DOUBLE, &count);
        double *recvbuff;
        recvbuff = (double*)malloc(sizeof(double)*count);
        MPI_Recv(recvbuff, count, MPI_DOUBLE, probe_status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
        if(count == 6)
        { 
          rank_bounds[6*probe_status.MPI_SOURCE + 0] = recvbuff[0]; 
          rank_bounds[6*probe_status.MPI_SOURCE + 1] = recvbuff[1];
          rank_bounds[6*probe_status.MPI_SOURCE + 2] = recvbuff[2];
          rank_bounds[6*probe_status.MPI_SOURCE + 3] = recvbuff[3];
          rank_bounds[6*probe_status.MPI_SOURCE + 4] = recvbuff[4];
          rank_bounds[6*probe_status.MPI_SOURCE + 5] = recvbuff[5];
          
          allReceived[recv_status.MPI_SOURCE] = true;
        }
        else
        {
          std::cout << "[" << rank << "] Corrupt message received from " << probe_status.MPI_SOURCE << std::endl;
        }
      }
      MPI_Buffer_detach(&buf_1, &bufsize_1);
      free(buf_1);
      MPI_Barrier(MPI_COMM_WORLD);

/*      if(rank == 0)
      {
        std::cout << "[" << rank << "] Bounding Boxes:" << std::endl;
        for(int i = 0; i < num_ranks; i++)
        {
      std::cout << "[" << i << "] " << rank_bounds[i*6 + 0] << "," << rank_bounds[i*6 + 1] << "," 
                << rank_bounds[i*6 + 2] << "," << rank_bounds[i*6 + 3] << "," << rank_bounds[i*6 + 4] << "," << rank_bounds[i*6 + 5] << std::endl; 
        }
      }*/
/****  Done - Setting up bounding box and neighborhood information to evaluate containing domain  ****/

/****  Creating initial list of particles  ****/
      UpdateSeedResolution(dom);
      InitializeUniformSeeds(dom, rank);
/****  Done - Creating initial list of particles  ****/
    } // End cycle = 0 initial set up. 
    
    cycle += 1;


/**** Start Particle Advection ****/
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> basisParticleArray, ex_basisParticleArray;
    basisParticleArray = vtkm::cont::make_ArrayHandle(next); 
    
    if(ex_next.size() > 0)
    {
      ex_basisParticleArray = vtkm::cont::make_ArrayHandle(ex_next);
    }

    using FieldHandle = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3>>;
    FieldHandle field; // = dom.GetField(m_field_name).GetData().Cast<FieldHandle>();
    dom.GetField(m_field_name).GetData().CopyTo(field);
    const vtkm::cont::DynamicCellSet& cells = dom.GetCellSet(0);
    const vtkm::cont::CoordinateSystem& coords = dom.GetCoordinateSystem();
    vtkm::Bounds bounds = dom.GetCoordinateSystem().GetBounds();
    using AxisHandle = vtkm::cont::ArrayHandle<vtkm::FloatDefault>;
    using RectilinearType = vtkm::cont::ArrayHandleCartesianProduct<AxisHandle, AxisHandle, AxisHandle>;
    using UniformType = vtkm::cont::ArrayHandleUniformPointCoordinates;

    vtkm::worklet::ParticleAdvection particleadvection;
    vtkm::worklet::ParticleAdvectionResult res, ex_res;

    if (coords.GetData().IsType<RectilinearType>())
    {
      using RectilinearGridEvalType = vtkm::worklet::particleadvection::RectilinearGridEvaluate<FieldHandle>;
      using RK4IntegratorType = vtkm::worklet::particleadvection::RK4Integrator<RectilinearGridEvalType>;
      RectilinearGridEvalType eval(coords, cells, field);
      RK4IntegratorType rk4(eval, static_cast<vtkm::Float32>(this->m_step_size));
      res = particleadvection.Run(rk4, basisParticleArray, 1); // Taking a single step

      if(ex_next.size() > 0)
      {
        ex_res = particleadvection.Run(rk4, ex_basisParticleArray, 1);
      }
    }
    else if (coords.GetData().IsType<UniformType>())
    {
      using UniformGridEvalType = vtkm::worklet::particleadvection::UniformGridEvaluate<FieldHandle>;
      using RK4IntegratorType = vtkm::worklet::particleadvection::RK4Integrator<UniformGridEvalType>;
      UniformGridEvalType eval(coords, cells, field);
      RK4IntegratorType rk4(eval, static_cast<vtkm::Float32>(this->m_step_size));
      res = particleadvection.Run(rk4, basisParticleArray, 1); // Taking a single step
      if(ex_next.size() > 0)
      {
        ex_res = particleadvection.Run(rk4, ex_basisParticleArray, 1);
      }
    }
    else
    {
      std::cout << "Data set type is not rectilinear or uniform." << std::endl; 
    }

// Update locations of particles after advection step. 

    auto particle_positions = res.positions;
    auto particle_stepstaken = res.stepsTaken;
   
    auto end_position = particle_positions.GetPortalControl();
    auto portal_stepstaken = particle_stepstaken.GetPortalControl();
  
    for (vtkm::Id index = 0; index < res.positions.GetNumberOfValues(); index++)
    {
      auto end_point = end_position.Get(index);
      auto steps = portal_stepstaken.Get(index);
     
      next[index] = vtkm::Vec<vtkm::FloatDefault, 3>((float)end_point[0],(float)end_point[1],(float)end_point[2]);
    } 

    if(ex_next.size() > 0)
    {
      auto ex_particle_positions = ex_res.positions;
      auto ex_particle_stepstaken = ex_res.stepsTaken;
      
      auto ex_end_position = ex_particle_positions.GetPortalControl();
      auto ex_portal_stepstaken = ex_particle_stepstaken.GetPortalControl();
     
      for (vtkm::Id index = 0; index < ex_res.positions.GetNumberOfValues(); index++)
      {
        auto end_point = ex_end_position.Get(index);
        auto steps = ex_portal_stepstaken.Get(index);
   
        ex_next[index] = vtkm::Vec<vtkm::FloatDefault, 3>((float)end_point[0],(float)end_point[1],(float)end_point[2]);
      }
    }
 
 /**** Done - advecting particle ****/
 
 /**** Exchange seed particles ****/
    ExchangeSeedParticles(rank, num_ranks);   
    if(cycle % this->m_write_frequency == 0)
    {
      ExchangeParticleInformation(rank, num_ranks);
      WriteBasisFlowInformation(rank, num_ranks); 
      ResetSeedParticles(dom, rank);
    }
 
    m_output->AddDomain(dom, domain_id);
//    delete min_dom;
  } // End else - num_domain == 1

#endif
}

inline bool Lagrangian::BoundsCheck(float x, float y, float z, double *BB)
{
  if(x >= BB[0] && x <= BB[1] && y >= BB[2] && y <= BB[3] && z >= BB[4] && z <= BB[5])
  {
    return true;
  }
  return false;
}

inline bool Lagrangian::BoundsCheck(float x, float y, float z, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
{
  if(x >= xmin && x <= xmax && y >= ymin && y <= ymax && z >= zmin && z <= zmax)
  {
    return true;
  }
  return false;
}


inline bool Lagrangian::AllMessagesReceived(bool *a, int num_ranks)
{
for(int i = 0; i < num_ranks; i++)
{
  if(a[i] == false)
    return false;
}
return true;
}

inline void Lagrangian::InitializeUniformSeeds(vtkm::cont::DataSet dom, vtkm::Id rank)
{
  vtkm::Bounds b = dom.GetCoordinateSystem().GetBounds();
  double dX = b.X.Max - b.X.Min;
  double dY = b.Y.Max - b.Y.Min;
  double dZ = b.Z.Max - b.Z.Min;

  double incX = dX/(seed_dims[0] - 1);
  double incY = dY/(seed_dims[1] - 1);
  double incZ = dZ/(seed_dims[2] - 1);

  vtkm::Id seed_cnt = maxID;
  
  for(int x = 0; x < seed_dims[0]-1; x++)
  {
    for(int y = 0; y < seed_dims[1]-1; y++)
    {
      for(int z = 0; z < seed_dims[2]-1; z++)
      {
          initpts.push_back(vtkm::Vec<vtkm::FloatDefault, 3>(b.X.Min + 0.5*incX +  x*incX,
          b.Y.Min + 0.5*incY + y*incY, b.Z.Min + 0.5*incZ + z*incZ));
          seedValidity.push_back(vtkm::Id(1));
          seedID.push_back(seed_cnt);
          seedRank.push_back(rank);
          seed_cnt++;
       }  
    }  
  } 
  
  maxID = seed_cnt; // This is the value of the smallest un-assigned ID.
  next.assign(initpts.begin(), initpts.end());
}

inline void Lagrangian::UpdateSeedResolution(const vtkm::cont::DataSet input)
{ 
  vtkm::cont::DynamicCellSet cell_set = input.GetCellSet();
  
  if (cell_set.IsSameType(vtkm::cont::CellSetStructured<1>()))
  { 
    vtkm::cont::CellSetStructured<1> cell_set1 = cell_set.Cast<vtkm::cont::CellSetStructured<1>>();
    vtkm::Id dims1 = cell_set1.GetPointDimensions();
    seed_dims[0] = dims1;
    if (this->m_cust_res)
    { 
      seed_dims[0] = dims1 / this->m_x_res;
    }
  }
  else if (cell_set.IsSameType(vtkm::cont::CellSetStructured<2>()))
  { 
    vtkm::cont::CellSetStructured<2> cell_set2 = cell_set.Cast<vtkm::cont::CellSetStructured<2>>();
    vtkm::Id2 dims2 = cell_set2.GetPointDimensions();
    seed_dims[0] = dims2[0];
    seed_dims[1] = dims2[1];
    if (this->m_cust_res)
    { 
      seed_dims[0] = dims2[0] / this->m_x_res;
      seed_dims[1] = dims2[1] / this->m_y_res;
    }
  }
  else if (cell_set.IsSameType(vtkm::cont::CellSetStructured<3>()))
  { 
    vtkm::cont::CellSetStructured<3> cell_set3 = cell_set.Cast<vtkm::cont::CellSetStructured<3>>();
    vtkm::Id3 dims3 = cell_set3.GetPointDimensions();
    seed_dims[0] = dims3[0];
    seed_dims[1] = dims3[1];
    seed_dims[2] = dims3[2];
    if (this->m_cust_res)
    { 
      seed_dims[0] = dims3[0] / this->m_x_res;
      seed_dims[1] = dims3[1] / this->m_y_res;
      seed_dims[2] = dims3[2] / this->m_z_res;
    }
  }
}

void Lagrangian::ExchangeSeedParticles(vtkm::Id rank, vtkm::Id num_ranks)
{
#ifdef VTKH_PARALLEL
  int num_particles_to_node[num_ranks] = {0};

  for(int i = 0; i < next.size(); i++)
  {
    if(!BoundsCheck(next[i][0], next[i][1], next[i][2], BB) && seedValidity[i] == 1 && seedRank[i] == rank)
    {
      int flag = 0;
      for(int n = 0; n < num_ranks; n++)
      {
        if(BoundsCheck(next[i][0], next[i][1], next[i][2],
            rank_bounds[n*6+0], rank_bounds[n*6+1], rank_bounds[n*6+2],
            rank_bounds[n*6+3], rank_bounds[n*6+4], rank_bounds[n*6+5]))
        {
          flag = 1;
          seedRank[i] = n; 
          num_particles_to_node[n] += 1;
        }
      }
      if(flag  == 0)
      {
        seedValidity[i] = 0;
      }
    }
  }

  for(int i = 0; i < ex_ID.size(); i++)
  {
    if(!BoundsCheck(ex_next[i][0], ex_next[i][1], ex_next[i][2], BB) && ex_valid[i] == 1 && ex_rank[i] == rank)
    {
      int flag = 0;
      for(int n = 0; n < num_ranks; n++)
      {
        if(BoundsCheck(ex_next[i][0], ex_next[i][1], ex_next[i][2],
            rank_bounds[n*6+0], rank_bounds[n*6+1], rank_bounds[n*6+2],
            rank_bounds[n*6+3], rank_bounds[n*6+4], rank_bounds[n*6+5]))
        {
          ex_rank[i] = n;
          num_particles_to_node[n] += 1;
        }
      }
      if(flag == 0)
      {
        ex_valid[i] = 0;
      }
    }
  }

  bool allReceived[num_ranks] = {false};
  allReceived[rank] = true;

  int ierr;
  int bufsize_2 = (next.size() + ex_ID.size())*20 + (MPI_BSEND_OVERHEAD * num_ranks);
  double *buf_2 = (double*)malloc(sizeof(double)*bufsize_2);
  MPI_Buffer_attach(buf_2, bufsize_2);

  // SEND LOOP
  for(int n = 0; n < num_ranks; n++)
  { 
    if(n != rank)
    {
      int buffsize = num_particles_to_node[n]*5;
      double *sendbuff;
      
      if(buffsize == 0)
        sendbuff = (double*)malloc(sizeof(double));
      else
        sendbuff = (double*)malloc(sizeof(double)*buffsize);
      
      MPI_Request send_request;
      if(buffsize == 0)
      { 
        buffsize = 1; 
        sendbuff[0] = 0.0;
      }
      else
      { 
        int count = 0; 
        for(int p = 0; p < next.size(); p++)
        { 
          if(seedRank[p] == n && seedValidity[p] == 1)
          { 
            sendbuff[count*5+0] = seedID[p] * 1.0;
            sendbuff[count*5+1] = next[p][0];
            sendbuff[count*5+2] = next[p][1];
            sendbuff[count*5+3] = next[p][2];
            sendbuff[count*5+4] = rank * 1.0;
            seedValidity[p] = 0;
            count++;
          }
        }
        for(int p = 0; p < ex_next.size(); p++)
        { 
          if(ex_rank[p] == n && ex_valid[p] == 1)
          { 
            sendbuff[count*5+0] = ex_ID[p] * 1.0;
            sendbuff[count*5+1] = ex_next[p][0];
            sendbuff[count*5+2] = ex_next[p][1];
            sendbuff[count*5+3] = ex_next[p][2]; 
            sendbuff[count*5+4] = ex_orig[p] * 1.0;
            ex_valid[p] = 0;
            count++;
          }
        }
      }

      ierr = MPI_Bsend(sendbuff, buffsize, MPI_DOUBLE, n, 0, MPI_COMM_WORLD); //, &send_request);
      free(sendbuff);
    }
  }
   // MPI_Probe, MPI_Recv Loop
  while(!AllMessagesReceived(allReceived, num_ranks))
  {
    MPI_Status probe_status, recv_status, wait_status;
    ierr = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &probe_status);
    int count;
    MPI_Get_count(&probe_status, MPI_DOUBLE, &count);
    double *recvbuff;
    recvbuff = (double*)malloc(sizeof(double)*count);
    MPI_Recv(recvbuff, count, MPI_DOUBLE, probe_status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
    if(count == 1)
    {
      // Rank probe_status.MPI_SOURCE has no particles to send.
      allReceived[recv_status.MPI_SOURCE] = true;
    }
    else if(count % 5 == 0)
    {
      int num_particles = count/5;

      for(int i = 0; i < num_particles; i++)
      {
        ex_ID.push_back((vtkm::Id)recvbuff[i*5+0]);
        ex_next.push_back(vtkm::Vec<vtkm::FloatDefault, 3>(recvbuff[i*5+1],recvbuff[i*5+2],recvbuff[i*5+3]));
        ex_orig.push_back((vtkm::Id)recvbuff[i*5+4]);
        ex_valid.push_back(1);
        ex_rank.push_back(rank);
      }
      allReceived[recv_status.MPI_SOURCE] = true;
    }
    else
    {
      std::cout << "Rank " << rank << " recevied message of incorrect length from rank : " << recv_status.MPI_SOURCE << std::endl;
      allReceived[recv_status.MPI_SOURCE] = true;
    }
    free(recvbuff);
  }

  MPI_Buffer_detach(&buf_2, &bufsize_2);
  free(buf_2);
  MPI_Barrier(MPI_COMM_WORLD);

#endif
}

void Lagrangian::ExchangeParticleInformation(vtkm::Id rank, vtkm::Id num_ranks)
{
#ifdef VTKH_PARALLEL
  int num_particles_to_node[num_ranks] = {0};
  for(int i = 0 ; i < ex_ID.size(); i++)
  { 
    if(ex_rank[i] == rank && ex_valid[i] == 1)
    { 
      if(ex_orig[i] == rank)
      { 
        // Particle is already back home. No need to do anything really. Use the particles ID to validate it and update the value of nextx..
        for(int n = 0 ; n < seedValidity.size(); n++)
        { 
          if(ex_ID[i] == seedID[n])
          { 
            if(seedValidity[n] == 0) // We had previously invalidated the seed. Verify that. Don't modify a valid seed either. 
            { 
              seedValidity[n] = 1;
              seedRank[n] = rank;
              next[n][0] = ex_next[i][0];
              next[n][1] = ex_next[i][1];
              next[n][2] = ex_next[i][2];
            }
          }
        }
      }
      else
      { // ex_orig is not the same rank.
        if(ex_orig[i] < num_ranks && ex_orig[i] >= 0)
          num_particles_to_node[ex_orig[i]] += 1;
      }
    }
  }
  bool allReceived[num_ranks] = {false};
  allReceived[rank] = true;

/*
  std::cout << "[" << rank << "] ";
  for(int i = 0; i < num_ranks; i++)
  {
    std::cout << num_particles_to_node[i] << " " ;
  }
  std::cout << std::endl;
*/

  int ierr;
  int bufsize_3 = (next.size() + ex_ID.size())* 20 + (MPI_BSEND_OVERHEAD + 1) * num_ranks; 
  double *buf_3 = (double*)malloc(sizeof(double)*bufsize_3);
  MPI_Buffer_attach(buf_3, bufsize_3);

  for(int n = 0; n < num_ranks; n++)
  {
    if(n != rank)
    {
      int buffsize = num_particles_to_node[n]*5;
      double *sendbuff;

      if(buffsize == 0)
        sendbuff = (double*)malloc(sizeof(double));
      else
        sendbuff = (double*)malloc(sizeof(double)*buffsize);

      MPI_Request send_request;
      MPI_Status send_status;
      if(buffsize == 0)
      {
        buffsize = 1;
        sendbuff[0] = 0.0;
      }
      else
      {
        int count = 0;
        for(int p = 0; p < ex_ID.size(); p++)
        {
          if(ex_orig[p] == n && ex_valid[p] == 1 && ex_rank[p] == rank && buffsize >= (count*5 + 5))
          {
            sendbuff[count*5+0] = ex_ID[p];
            sendbuff[count*5+1] = ex_next[p][0];
            sendbuff[count*5+2] = ex_next[p][1];
            sendbuff[count*5+3] = ex_next[p][2];
            sendbuff[count*5+4] = ex_orig[p];
            ex_valid[p] = 0;
            count++;
          }
        }
      }
      
      ierr = MPI_Bsend(sendbuff, buffsize, MPI_DOUBLE, n, 0, MPI_COMM_WORLD); //, &send_request);
//      MPI_Request_free(&send_request);
//      MPI_Wait(&send_request, &send_status);
      free(sendbuff);
    }
  }

  int num_particles = 0;
  while(!AllMessagesReceived(allReceived, num_ranks))
  {
    MPI_Status probe_status, recv_status;
    MPI_Request request;
    ierr = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &probe_status);
    int count;
    MPI_Get_count(&probe_status, MPI_DOUBLE, &count);
    double *recvbuff;
    recvbuff = (double*)malloc(sizeof(double)*count);
    MPI_Recv(recvbuff, count, MPI_DOUBLE, probe_status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
//    MPI_Irecv(recvbuff, count, MPI_DOUBLE, probe_status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
//    MPI_Wait(&request,&recv_status);
    if(count == 1)
    {
      // Rank probe_status.MPI_SOURCE has no particles to send.
      allReceived[recv_status.MPI_SOURCE] = true;
    }
    else if(count % 5 == 0)
    {
      num_particles = count/5;
      for(int i = 0; i < num_particles; i++)
      {
        vtkm::Id id = recvbuff[i*5+0];
        for(int n = 0 ; n < seedValidity.size(); n++)
        {
          if(id == seedID[n])
          {
            if(seedValidity[n] == 0) // We had previously invalidated the seed. Verify that. Don't modify a valid seed either. 
            {
              seedValidity[n] = 1;
              next[n][0] = recvbuff[i*5+1];
              next[n][1] = recvbuff[i*5+2];
              next[n][2] = recvbuff[i*5+3];
              seedRank[n] = recvbuff[i*5+4];
            }
          }
        }
      }
      allReceived[recv_status.MPI_SOURCE] = true;
    }
    else
    {
      std::cout << "Rank " << rank << " received message of incorrect length from rank : " << recv_status.MPI_SOURCE << std::endl;
      allReceived[recv_status.MPI_SOURCE] = true;
    }
    free(recvbuff);
  }
  MPI_Buffer_detach(&buf_3, &bufsize_3);
  free(buf_3);
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void Lagrangian::WriteBasisFlowInformation(vtkm::Id rank, vtkm::Id num_ranks)
{
  int connectivity_index = 0;
  std::vector<vtkm::Id> connectivity;
  std::vector<vtkm::Vec<vtkm::Float32, 3>> pointCoordinates;
  std::vector<vtkm::UInt8> shapes;
  std::vector<vtkm::IdComponent> numIndices;
  std::vector<vtkm::Id> basisFlowId;
  int num_seeds = seedValidity.size();
  for(int n = 0; n < num_seeds; n++)
  { 
    if(seedValidity[n] == 1 && seedRank[n] == rank)
    { 
      connectivity.push_back(connectivity_index); 
      connectivity.push_back(connectivity_index + 1);
      connectivity_index += 2;
      pointCoordinates.push_back(vtkm::Vec<vtkm::Float32,3>(initpts[n][0], initpts[n][1], initpts[n][2]));
      pointCoordinates.push_back(vtkm::Vec<vtkm::Float32,3>(next[n][0], next[n][1], next[n][2]));
      shapes.push_back(vtkm::CELL_SHAPE_LINE);
      numIndices.push_back(2);
      basisFlowId.push_back(seedID[n]);
    }
  }
  vtkm::cont::DataSetBuilderExplicit full_dsbe;
   
  vtkm::cont::DataSet full_dataset = full_dsbe.Create(pointCoordinates, shapes, numIndices, connectivity);
  vtkm::cont::DataSetFieldAdd full_dsfa;
  full_dsfa.AddCellField(full_dataset, "ID", basisFlowId);
  
  std::stringstream full_s;
  full_s << "output/basisflows_" << rank << "_" << cycle << ".vtk";
    
  vtkm::io::writer::VTKDataSetWriter writer(full_s.str().c_str());
  writer.WriteDataSet(full_dataset);
}

void Lagrangian::ResetSeedParticles(vtkm::cont::DataSet dom, vtkm::Id rank)
{
  seedValidity.clear();
  seedID.clear();
  seedRank.clear();
  next.clear();
  initpts.clear();
  ex_next.clear();
  ex_valid.clear();
  ex_rank.clear();
  ex_orig.clear();
  ex_ID.clear();
  InitializeUniformSeeds(dom, rank);
}

std::string
Lagrangian::GetName() const
{
  return "vtkh::Lagrangian";
}

} //  namespace vtkh
