#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

// Utility to dump out an lmdb archive. A way to check images.
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

// Boost
#include "boost/scoped_ptr.hpp"

// OpenCV
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"

// LArCV
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"



int main( int nargs, char** argv ) {

  std::string input_binproto = argv[1];
  std::string output_rootfile = argv[2];
  int crop_timepad = 10;
  int fudgeoffset = -1;
  
  // Image protobuf
  caffe::BlobProto datum;

  // Output IOManager
  larcv::IOManager ioman(larcv::IOManager::kWRITE, "IOManager");
  ioman.set_out_file( output_rootfile );
  ioman.initialize();

  std::ifstream mean_file( input_binproto.c_str(), std::ios::binary );

  std::ostringstream buffer;
  buffer << mean_file.rdbuf();

  datum.ParseFromString( buffer.str() );


  auto event_images = (larcv::EventImage2D*)ioman.get_data( larcv::kProductImage2D, "tpc_12ch_mean" );
  //auto event_roi    = (larcv::EventROI*)ioman.get_data( larcv::kProductROI, "tpc_12ch_mean" );

  //const std::string& data = datum.data();
  //std::vector<char> vec_data( data.c_str(), data.c_str()+data.size());
  int height = datum.height();
  int cropped_height = height-2*crop_timepad;
  int width = datum.width();
  int nchannels = datum.channels();

  std::cout << "Mean image: " << height << " x " << width << " x " << nchannels << std::endl;
    
  // tpc layers
  for (int c=0; c<nchannels; c++) { 
    larcv::ImageMeta meta( 3456, 4608, width, cropped_height, 0, 4608, c/4 );
    larcv::Image2D img2d( meta );
    for (int h=0; h<cropped_height; h++) {
      for (int w=0; w<width; w++) {
	int index = (c*(height) + (h+crop_timepad+fudgeoffset))*width + w;
	float val = datum.data(index);
	img2d.set_pixel( w, h, (float)val );
      }
    }
    event_images->Append( img2d );
  }
    
  ioman.set_id( 0, 0, 0 );
  ioman.save_entry();
  ioman.finalize();

  std::cout << "FIN." << std::endl;
}
