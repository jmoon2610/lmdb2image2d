#include <iostream>
#include <string>

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


void parselist( std::string fname, std::vector<std::string>& vec ) {
  std::ifstream f( fname.c_str() );

  while ( !f.eof() && f.good() ) {
    char buffer[2054];
    f >> buffer;
    std::string sbuf = buffer;
    if (sbuf!="") {
      if ( vec.size()==0 || vec.at(vec.size()-1)!=std::string(sbuf) )
	vec.push_back( sbuf );
    }
  }
  f.close();
}

int main( int nargs, char** argv ) {

  std::string input_lmdb = argv[1];
  std::string output_rootdb = argv[2];
  std::string selection_file = "__none__";
  if (nargs==4) {
    selection_file = argv[3];
  }

  std::string FLAGS_backend = "lmdb";
  bool util_makecut   = false;

  int nprocess = 10;
  int crop_timepad = 10;

  boost::scoped_ptr<caffe::db::DB> db(caffe::db::GetDB(FLAGS_backend));
  db->Open( input_lmdb.c_str(), caffe::db::READ );
  boost::scoped_ptr<caffe::db::Cursor> cursor(db->NewCursor());

  cv::Scalar r(200,0,0);
  cv::Scalar g(0,200,0);
  cv::Scalar b(0,0,200);
  cv::Scalar color[3] = { r, g, b};

  std::vector<std::string> keylist;
  if ( util_makecut && nargs==4 ) {
    parselist( selection_file, keylist );
  }

  
  // Image protobuf
  caffe::Datum datum;

  // Output IOManager
  larcv::IOManager ioman(larcv::IOManager::kWRITE, "IOManager");
  ioman.set_out_file( output_rootdb );
  ioman.initialize();

  int nimages  = 0;

  while ( cursor->valid() ) {

    if ( util_makecut ) {
      bool foundit = false;
      for ( std::vector<std::string>::iterator it=keylist.begin(); it!=keylist.end(); it++ )  {
	if ( cursor->key()==(*it) ) {
	  std::cout << "Found image" << std::endl;
	  foundit = true;
	  break;
	}
      }
      if ( !foundit ) {
	cursor->Next();
	continue;
      }
    }

    datum.ParseFromString( cursor->value() );
    std::cout << "[ label " << datum.label() << "] key=" << cursor->key() << " " << datum.width() << " x " << datum.height() << " x " << datum.channels() << std::endl;

    auto event_images = (larcv::EventImage2D*)ioman.get_data( larcv::kProductImage2D, "tpc_12ch" );
    auto event_roi    = (larcv::EventROI*)ioman.get_data( larcv::kProductROI, "tpc_12ch" );


    const std::string& data = datum.data();
    std::vector<char> vec_data( data.c_str(), data.c_str()+data.size());
    int height = datum.height();
    int cropped_height = height-2*crop_timepad;
    int width = datum.width();
    int nchannels = datum.channels();
    
    // tpc layers
    for (int c=0; c<nchannels; c++) { 
      larcv::ImageMeta meta( 3456, 4608, width, cropped_height, 0, 4608, c/4 );
      larcv::Image2D img2d( meta );
      for (int h=0; h<cropped_height; h++) {
	for (int w=0; w<width; w++) {
	  int index = (c*(height) + (h+crop_timepad))*width + w;
	  unsigned int val = static_cast<unsigned short>( vec_data.at(index) );
	  img2d.set_pixel( w, h, (float)val );
	}
      }
      event_images->Append( img2d );
    }
    
    // parse key
    std::string key = cursor->key();
    size_t found = key.find_first_of("_");
    int keyid = std::atoi(key.substr(0,found).c_str());
    ioman.set_id( 0, 0, keyid );
    ioman.save_entry();
    
    
    cursor->Next();
    nimages++;
    if (nprocess>0 && nimages>=nprocess)
      break;
  }
  ioman.finalize();

  std::cout << "FIN." << std::endl;
}
