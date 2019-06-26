#ifndef __GZIP_H__
#define __GZIP_H__

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <sstream>

// A wraper class that uses boost gzip library to compress the file.
class Gzip {
public:
  static std::string compress(const std::string &data) {
    namespace bio = boost::iostreams;

    std::stringstream compressed;
    std::stringstream origin(data);

    bio::filtering_streambuf<bio::input> out;
    out.push(bio::gzip_compressor(bio::gzip_params(bio::gzip::best_speed)));
    out.push(origin);
    bio::copy(out, compressed);

    return compressed.str();
  }

  static std::string decompress(const std::string &data) {
    namespace bio = boost::iostreams;

    std::stringstream compressed(data);
    std::stringstream decompressed;

    bio::filtering_streambuf<bio::input> out;
    out.push(bio::gzip_decompressor());
    out.push(compressed);
    bio::copy(out, decompressed);

    return decompressed.str();
  }
};

#endif // __GZIP_H__
