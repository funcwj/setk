// rir-generator.h
// wujian@18.6.26

#ifndef RIR_GENERATOR_H
#define RIR_GENERATOR_H

#include <map>
#include <iterator>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

typedef enum {
    kBidirectional,
    kHypercardioid,
    kCardioid,
    kSubcardioid,
    kOmnidirectional
} PolorPattern;

// Plan to use this struct
// struct Point3D {
//     BaseFloat x, y, z;
//     Point3D(): x(0), y(0), z(0) {}
//     Point3D(BaseFloat x, BaseFloat y, BaseFloat z): x(x), y(y), z(z) {}
// };


struct RirGeneratorOptions {
    std::string source_location, receiver_location, 
                room_topo, orientation, beta, microphone_type;
    BaseFloat sound_velocity, samp_frequency;
    bool hp_filter;
    int32 num_samples, order;

    RirGeneratorOptions(): sound_velocity(340), samp_frequency(16000),
                    hp_filter(true), num_samples(-1), order(-1),
                    microphone_type("omnidirectional"), source_location(""),
                    receiver_location(""), room_topo(""), orientation(""), 
                    beta("") {}

    void Register(OptionsItf *opts) {
        opts->Register("sound-velocity", &sound_velocity, "Sound velocity in m/s");
        opts->Register("samp-frequency", &samp_frequency, "Sampling frequency in Hz");
        opts->Register("hp-filter", &hp_filter, "If true, high-pass filter is enabled");
        opts->Register("number-samples", &num_samples, "Number of samples to calculate");
        opts->Register("order", &order, "Reflection order, default is -1, i.e. maximum order");
        opts->Register("microphone-type", &microphone_type, "Type of micrphone arrays("
                        "\"omnidirectional\"|\"subcardioid\"|\"cardioid\"|\"hypercardioid\"|\"bidirectional\")");
        opts->Register("receiver-location", &receiver_location, "3D-Coordinates of receivers in .m. "
                        "Each coordinate is separated by a single semicolon, egs: --receiver-location=2,1.5,2;1,1.5,2");
        opts->Register("source-location", &source_location, "3D Coordinates of receivers in .m. "
                        "egs: --source-location=2,3.5,2");
        opts->Register("room-topo", &room_topo, "Room dimensions in .m. egs: --room-dim=5,4,6");
        opts->Register("angle", &orientation, "Direction in which the microphones are pointed, "
                        "specified using azimuth and elevation angles(in radians)");
        opts->Register("beta", &beta, "6D vector specifying the reflection coefficients or reverberation time(T60) in seconds.");
    }

};

class RirGenerator {

public:
    RirGenerator(RirGeneratorOptions &opts): 
            opts_(opts), velocity_(opts.sound_velocity),
            frequency_(opts.samp_frequency), hp_filter_(opts.hp_filter),
            num_samples_(opts.num_samples), order_(opts.order) {
        ComputeDerived();
    }

    void GenerateRir(Matrix<BaseFloat> *rir);

    std::string Report();

    BaseFloat Frequency() { return frequency_; }

private:
    BaseFloat velocity_, frequency_, revb_time_;
    bool hp_filter_;
    int32 num_samples_, order_, room_dim_, num_mics_;

    RirGeneratorOptions opts_;
    
    std::vector<std::vector<BaseFloat> > receiver_location_;
    std::vector<BaseFloat> source_location_, room_topo_, angle_, beta_;

    std::map<std::string, PolorPattern> str_to_pattern_ = {
        {"omnidirectional", kOmnidirectional},
        {"subcardioid", kSubcardioid},
        {"cardioid", kCardioid},
        {"hypercardioid", kHypercardioid},
        {"bidirectional", kBidirectional}
    };

    void ComputeDerived();

    BaseFloat MicrophoneSim(BaseFloat x, BaseFloat y, BaseFloat z);
};


double Sinc(double x) {
    return x == 0 ? 1.0 : std::sin(x) / x;
}

BaseFloat Sabine(std::vector<BaseFloat> &room_topo, std::vector<BaseFloat> &beta, BaseFloat c) {
    BaseFloat V = room_topo[0] * room_topo[1] * room_topo[2];
    BaseFloat alpha = ((1 - pow(beta[0], 2)) + (1 - pow(beta[1], 2))) * room_topo[1] * room_topo[2] +
                ((1 - pow(beta[2], 2)) + (1 - pow(beta[3], 2))) * room_topo[0] * room_topo[2] +
                ((1 - pow(beta[4], 2)) + (1 - pow(beta[5], 2))) * room_topo[0] * room_topo[1];
    BaseFloat revb_time = 24 * Log(10.0) * V / (c * alpha);
    return std::max(0.128f, revb_time);
}


} // namespace kaldi

#endif