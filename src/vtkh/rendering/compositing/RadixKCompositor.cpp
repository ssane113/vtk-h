#include <vtkh/rendering/ImageCompositor.hpp>
#include "RadixKCompositor.hpp"
#include "vtkh_diy_collect.hpp"
#include "vtkh_diy_utils.hpp"

#include <diy/master.hpp>
#include <diy/mpi.hpp>
#include <diy/partners/swap.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>

// debug includes
#include <iostream>
#include <sstream>

namespace vtkh
{

void reduce_images(void *b, 
                   const diy::ReduceProxy &proxy,
                   const diy::RegularSwapPartners &partners) 
{
  ImageBlock *block = reinterpret_cast<ImageBlock*>(b);
  unsigned int round = proxy.round();
  Image &image = block->m_image; 
  // count the number of incoming pixels
  if(proxy.in_link().size() > 0)
  {
      for(int i = 0; i < proxy.in_link().size(); ++i)
      {
        int gid = proxy.in_link().target(i).gid;
        if(gid == proxy.gid())
        {
          //skip revieving from self since we sent nothing
          continue;
        }
        Image incoming; 
        proxy.dequeue(gid, incoming);
        vtkh::ImageCompositor compositor;
        //if(image.m_depths.size() != incoming.m_pixels.size() / 4 ||
        //  incoming.m_bounds.X.Min != image.m_bounds.X.Min || 
        //  incoming.m_bounds.Y.Min != image.m_bounds.Y.Min || 
        //  incoming.m_bounds.X.Max != image.m_bounds.X.Max || 
        //  incoming.m_bounds.Y.Max != image.m_bounds.Y.Max )
        //{
        //  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //  std::cout<<"image "<<rank<<" "; image.Print();
        //  std::cout<<"incoming "<< rank<<" "; incoming.Print();
        //}
        compositor.ZBufferComposite(image, incoming);
      } // for in links
  } 

  if(proxy.out_link().size() == 0)
  {
    return;
  }
  // do compositing?? intermediate stage?
  const int group_size = proxy.out_link().size(); 
  const int current_dim = partners.dim(round);
  
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::stringstream logname;
  logname<<"buckets_"<<rank<<"_rnd_"<<round<<"_dim_"<<current_dim<<".log";
  std::ofstream log;
  log.open(logname.str().c_str(), std::ofstream::out);
  log<<"Rank "<<rank<<" round "<<round<<" dim "<<current_dim<<"\n";
  //create balanced set of ranges for current dim
  diy::DiscreteBounds image_bounds = VTKMBoundsToDIY(image.m_bounds);
  // image bounds are inclusive
  int range_length = image_bounds.max[current_dim] - image_bounds.min[current_dim] + 1;
  int base_step = range_length / group_size;
  int rem = range_length % group_size;
  log<<"group size "<<group_size<<"\n";
  log<<"range length "<<range_length<<"\n";
  log<<"base step "<<base_step<<"\n";
  log<<"remainder "<<rem<<"\n";
  log<<"group members: ";
  for(int i = 0; i < group_size; ++i) log<<proxy.out_link().target(i).proc<<" ";
  log<<"\n";
  std::vector<int> bucket_sizes(group_size, base_step);
  for(int i  = 0; i < rem; ++i)
  {
    bucket_sizes[i]++;
    log<<"bucket "<<i<<" size "<<bucket_sizes[i]<<"\n";
  }

  int count = 0;
  for(int i  = 0; i < group_size; ++i)
  {
    count += bucket_sizes[i];
  }
  log<<"count "<<count<<" range length "<<range_length<<"\n";
  assert(count == range_length);
  log<<"Current image bounds "<<image.m_bounds<<"\n";
  std::vector<diy::DiscreteBounds> subset_bounds(group_size, VTKMBoundsToDIY(image.m_bounds));  
  int min_pixel = image_bounds.min[current_dim];
  for(int i = 0; i < group_size; ++i)
  {
    // ensure we create mutually exclusice bounds
    subset_bounds[i].min[current_dim] = min_pixel; 
    subset_bounds[i].max[current_dim] = min_pixel + bucket_sizes[i] - 1;
    log<<"group "<<i<<" min "<<subset_bounds[i].min[current_dim]<<" "; 
    log<<"max "<<subset_bounds[i].max[current_dim]<<"\n"; 
    min_pixel += bucket_sizes[i];

    log<<"group "<<i<<" ("<<subset_bounds[i].min[0]<<", "; 
    log<<subset_bounds[i].min[1]<<") - "; 
    log<<"("<<subset_bounds[i].max[0]<<","; 
    log<<subset_bounds[i].max[1]<<")\n"; 
    log<<"target id "<<proxy.out_link().target(i).proc<<"\n";
  }
  log.close();
  //debug
  if(group_size > 1)
  {
    for(int i = 1; i < group_size; ++i)
    {
      assert(subset_bounds[i-1].max[current_dim] + 1 == subset_bounds[i].min[current_dim]);
    }
  
    assert(subset_bounds[0].min[current_dim] == image_bounds.min[current_dim]);
    assert(subset_bounds[group_size-1].max[current_dim] == image_bounds.max[current_dim]);
  }
  
  std::vector<Image> out_images(group_size);
  for(int i = 0; i < group_size; ++i)
  {
    out_images[i].SubsetFrom(image, DIYBoundsToVTKM(subset_bounds[i]));  
  } //for

  for(int i = 0; i < group_size; ++i)
  {
      if(proxy.out_link().target(i).gid == proxy.gid())
      {
        image.Swap(out_images[i]);
      }
      else
      {
        proxy.enqueue(proxy.out_link().target(i), out_images[i]);
      }
  } //for 

} // reduce images

RadixKCompositor::RadixKCompositor()
{

}

RadixKCompositor::~RadixKCompositor()
{

}

void
RadixKCompositor::CompositeSurface(diy::mpi::communicator &diy_comm, Image &image)
{
  
    diy::DiscreteBounds global_bounds = VTKMBoundsToDIY(image.m_orig_bounds);

    // tells diy to use all availible threads
    const int num_threads = -1; 
    const int num_blocks = diy_comm.size(); 
    const int magic_k = 8;

    diy::Master master(diy_comm, num_threads);

    // create an assigner with one block per rank
    diy::ContiguousAssigner assigner(num_blocks, num_blocks); 
    AddImageBlock create(master, image);
    const int num_dims = 2;
    diy::RegularDecomposer<diy::DiscreteBounds> decomposer(num_dims, global_bounds, num_blocks);
    decomposer.decompose(diy_comm.rank(), assigner, create);
    diy::RegularSwapPartners partners(decomposer, 
                                      magic_k, 
                                      false); // false == distance halving
    diy::reduce(master,
                assigner,
                partners,
                reduce_images);


//    diy::all_to_all(master,
//                    assigner,
//                    CollectImages(decomposer),
//                    magic_k);
  
    if(diy_comm.rank() == 0) 
    {
      master.prof.output(m_timing_log);
    }
  
}

std::string 
RadixKCompositor::GetTimingString()
{
  std::string res(m_timing_log.str());
  m_timing_log.str("");
  return res;
}

} // namespace alpine
