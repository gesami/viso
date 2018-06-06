
#include <string>

#include "types.h"
#include "keyframe.h"

class FrameSequence
{
  public:
    class FrameHandler
    {
        virtual void OnNewFrame(Keyframe::Ptr keyframe) = 0;
    };

    FrameSequence(std::string location,
                  FrameHandler *handler) : location_(location),
                                           handler_(handler))
    {
    }

    void RunOnce()
    {
        bool success = true;
        std::string file = "000000";

        {
            std::string tmp = std::to_string(Keyframe::GetNextId());
            for (int i = 0; i < tmp.size(); ++i)
            {
                file[file.size() - tmp.size() + i] = tmp[i];
            }
        }

        file = file + ".png";

        cv::Mat frame = cv::imread(file, 0);
        success = (frame.data != NULL);

        if (success)
        {
            OnNewFrame(std::make_shared<Keyframe>(frame));
        }
    };

  private:
    std::string location_;
    FrameHandler *handler_;
};
