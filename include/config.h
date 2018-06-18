#ifndef CONFIG_H
#define CONFIG_H
// for cv
#include <opencv2/core/core.hpp>

// std
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>



using namespace std;




class Config
{
private:
    static std::shared_ptr<Config> config_; 
    cv::FileStorage file_;
    
    Config () {} // private constructor makes a singleton
public:
    ~Config();  // close the file when deconstructing 
    
    // set a new config file 
    static void setParameterFile( const std::string& filename ); 
    
    // access the parameter values
    template< typename T >
    static T get( const std::string& key )
    {
        return T( Config::config_->file_[key] );
    }
};


#endif // CONFIG_H