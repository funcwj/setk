// rir-generator.cc
// wujian@18.6.26

#include "include/rir-generator.h"

namespace kaldi {

void RirGenerator::ComputeDerived() {
    if (!str_to_pattern_.count(opts_.microphone_type))
        KALDI_ERR << "Unknown option values: --microphone-type=" << opts_.microphone_type;
    KALDI_ASSERT(frequency_ >= 1);
    KALDI_ASSERT(velocity_ >= 1);
    KALDI_ASSERT(order_ >= -1);

    // Process room topo
    KALDI_ASSERT(opts_.room_topo != "" && "Options --room-topo is not configured");
    KALDI_ASSERT(SplitStringToFloats(opts_.room_topo, ",", false, &room_topo_));
    room_dim_ = room_topo_.size();
    // KALDI_ASSERT(room_dim_ == 2 || room_dim_ == 3);
    KALDI_ASSERT(room_dim_ == 3);

    // Process source location
    KALDI_ASSERT(opts_.source_location != "" && "Options --source-location is not configured");
    KALDI_ASSERT(SplitStringToFloats(opts_.source_location, ",", false, &source_location_));
    std::vector<std::string> mics;

    // Process receiver_location
    KALDI_ASSERT(opts_.receiver_location != "" && "Options --receiver-location is not configured");
    SplitStringToVector(opts_.receiver_location, ";", false, &mics);
    num_mics_ = mics.size();
    receiver_location_.resize(num_mics_);
    for (int32 i = 0; i < num_mics_; i++)
        KALDI_ASSERT(SplitStringToFloats(mics[i], ",", false, &receiver_location_[i]));

    // Process angle
    std::vector<BaseFloat> angle_tmp;
    if (opts_.orientation == "") {
        for (int32 i = 0; i <= 1; i++)
            angle_tmp.push_back(0.0);
    } else {
        KALDI_ASSERT(SplitStringToFloats(opts_.orientation, ",", false, &angle_tmp));
    }
    if (angle_tmp.size() == 1)
        angle_tmp.push_back(0.0);
    KALDI_ASSERT(angle_tmp.size() == 2);
    angle_.swap(angle_tmp);

    // Process beta
    std::vector<BaseFloat> beta_tmp;
    KALDI_ASSERT(opts_.beta != "" && "Options --beta is not configured");
    KALDI_ASSERT(SplitStringToFloats(opts_.beta, ",", false, &beta_tmp));
    KALDI_ASSERT(beta_tmp.size() == 1 || beta_tmp.size() == 6);
    if (beta_tmp.size() == 1) {
        // beta_tmp[0] is T60
        revb_time_ = beta_tmp[0];
        BaseFloat V = room_topo_[0] * room_topo_[1] * room_topo_[2], S = 2 * (room_topo_[0] * room_topo_[1] 
                        + room_topo_[1] * room_topo_[2] + room_topo_[0] * room_topo_[2]);
        beta_.resize(6);
        if (revb_time_ != 0) {
            BaseFloat alfa = 24 * V * Log(10.0) / (velocity_ * S * revb_time_);
            if (alfa > 1) 
                KALDI_ERR << alfa << " > 1: The reflection coefficients cannot be calculated using the current"
                    << " room parameters, i.e. room size and reverberation time.";
            for (int32 i = 0; i < 6; i++)
                beta_[i] = std::sqrt(1 - alfa);
        } else {
            for (int32 i = 0; i < 6; i++)
                beta_[i] = 0;
        }
    } else {
        // compute from Sabine formula
        revb_time_ = Sabine(room_topo_, beta_tmp, velocity_);
    }
    beta_.swap(beta_tmp);

    // Process number of samples
    // if non-positive, compute from T60 
    if (!num_samples_) {
        num_samples_ = static_cast<int32>(revb_time_ * frequency_);
    }
    KALDI_ASSERT(num_samples_ && "Invalid number of samples");
}


void RirGenerator::GenerateRir(Matrix<BaseFloat> *rir) {
    rir->Resize(num_mics_, num_samples_);
    BaseFloat cts = velocity_ / frequency_;
    Vector<BaseFloat> R(3), S(3), T(3), Y(3);
    Vector<BaseFloat> Rp_plus_Rm(3), Rm(3), Refl(3);

    for (int32 i = 0; i < 3; i++) {
        S(i) = source_location_[i] / cts;
        T(i) = room_topo_[i] / cts;
    }

    BaseFloat dist, fdist, gain;
    int32 Tw = 2 * static_cast<int32>(0.004 * frequency_ + 0.5);
    Vector<BaseFloat> Lpi(Tw);
    BaseFloat W = 2 * M_PI * 100 / frequency_;
    BaseFloat R1 = exp(-W);
    BaseFloat B1 = 2 * R1 * cos(W);
    BaseFloat B2 = -R1 * R1;
    BaseFloat A1 = -1 - R1;

    for (int32 m = 0; m < num_mics_; m++) {
        for (int32 i = 0; i < 3; i++) 
            R(i) = receiver_location_[m][i] / cts;
        
        int32 nx = static_cast<int32>(ceil(num_samples_ / (2 * T(0))));
        int32 ny = static_cast<int32>(ceil(num_samples_ / (2 * T(1))));
        int32 nz = static_cast<int32>(ceil(num_samples_ / (2 * T(2))));

        for (int32 x = -nx; x <= nx; x++) {
            Rm(0) = 2 * x * T(0);

            for (int32 y = -ny; y <= ny; y++) {
                Rm(1) = 2 * y * T(1);

                for (int32 z = -nz; z <= nz; z++) {
                    Rm(2) = 2 * z * T(2);

                    for (int32 q = 0; q <= 1; q++) {
                        Rp_plus_Rm(0) = (1 - 2 * q) * S(0) - R(0) + Rm(0);
                        Refl(0) = pow(beta_[0], abs(x - q)) * pow(beta_[1], abs(x));

                        for (int32 j = 0; j <= 1; j++) {
                            Rp_plus_Rm(1) = (1 - 2 * j) * S(1) - R(1) + Rm(1);
                            Refl(1) = pow(beta_[2], abs(y - j)) * pow(beta_[3], abs(y));

                            for (int32 k = 0; k <= 1; k++) {
                                Rp_plus_Rm(2) = (1 - 2 * k) * S(2) - R(2) + Rm(2);
                                Refl(2) = pow(beta_[4], abs(z - k)) * pow(beta_[5], abs(z));

                                dist = sqrt(pow(Rp_plus_Rm(0), 2) + pow(Rp_plus_Rm(1), 2) + pow(Rp_plus_Rm(2), 2));
                                if (abs(2 * x - q) + abs(2 * y - j) + abs(2 * z - k) <= order_ || order_ == -1) {
                                    fdist = floor(dist);

                                    if (fdist < num_samples_) {
                                        gain = MicrophoneSim(Rp_plus_Rm(0), Rp_plus_Rm(1), Rp_plus_Rm(2))
                                            * Refl(0) * Refl(1) * Refl(2) / (4 * M_PI * dist * cts);
                                        
                                        for (int32 n = 0 ; n < Tw ; n++)
                                            Lpi(n) =  0.5 * (1 - cos(2 * M_PI * ((n + 1 - (dist - fdist)) / Tw))) * 
                                                Sinc(M_PI * (n + 1 - (dist - fdist) - (Tw / 2)));
                                        
                                        int32 pos = static_cast<int32>(fdist - (Tw / 2) + 1);
                                        for (int32 n = 0 ; n < Tw; n++)
                                            if (pos + n >= 0 && pos + n < num_samples_)
                                                (*rir)(m, pos + n) += gain * Lpi(n);
                                    }
                                }

                            }
                        }
                    }
                }
            }
        }

        if (hp_filter_) {
            Y.SetZero();
            BaseFloat X0;
            for (int32 i = 0; i < num_samples_; i++) {
                X0 = (*rir)(m, i);
                Y(2) = Y(1);
                Y(1) = Y(0);
                Y(0) = B1 * Y(1) + B2 * Y(2) + X0;
                (*rir)(m, i) = Y(0) + A1 * Y(1) + R1 * Y(2);
            }
        }
    }
}


BaseFloat RirGenerator::MicrophoneSim(BaseFloat x, BaseFloat y, BaseFloat z) {
    BaseFloat rho = 0;
    switch(str_to_pattern_[opts_.microphone_type]) {
        case kBidirectional:
            rho = 0;
            break;
        case kHypercardioid:
            rho = 0.25;
            break;
        case kCardioid:
            rho = 0.5;
            break;
        case kSubcardioid:
            rho = 0.75;
            break;
        case kOmnidirectional:
            rho = 1;
            break;
    }
    if (rho == 1)
        return 1;
    else {
        BaseFloat theta = acos(z / sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));
        BaseFloat phi = atan2(y, x);
        BaseFloat gain = sin(M_PI / 2 - angle_[1]) * sin(theta) * cos(angle_[0] - phi) + cos(M_PI / 2 - angle_[1]) * cos(theta);
        return rho + (1 - rho) * gain;
    }
}

std::string RirGenerator::Report() {
    std::ostringstream oss;
    oss << "RirGenerator Configures: " << std::endl;
    oss << "-- Sound Velocity: " << velocity_ << std::endl;
    oss << "-- Sample Frequency:  " << frequency_ << std::endl;
    oss << "-- Number of Samples: " << num_samples_ << std::endl;
    oss << "-- Order/Room Dim: " << order_ << "/" << room_dim_ << std::endl;
    oss << "-- PolarPattern: " << opts_.microphone_type << std::endl;
    oss << "-- Reverberation Time: " << revb_time_ << std::endl;
    oss << "-- Source Location: ( ";
    std::copy(source_location_.begin(), source_location_.end(), 
                std::ostream_iterator<float>(oss, " "));
    oss << ")" << std::endl;
    oss << "-- Room Topology: [ ";
    std::copy(room_topo_.begin(), room_topo_.end(), 
                std::ostream_iterator<float>(oss, " "));
    oss << "]" << std::endl;
    oss << "-- Angle: [ ";
    std::copy(angle_.begin(), angle_.end(), 
                std::ostream_iterator<float>(oss, " "));
    oss << "]" << std::endl;
    oss << "-- Beta Vector: [ ";
    std::copy(beta_.begin(), beta_.end(), 
                std::ostream_iterator<float>(oss, " "));
    oss << "]" << std::endl;
    oss << "-- Reciver Locations: ";
    for (int32 i = 0; i < receiver_location_.size(); i++) {
        oss << "( ";
        std::copy(receiver_location_[i].begin(), receiver_location_[i].end(), 
            std::ostream_iterator<float>(oss, " "));
        oss << ")";
    }
    oss << std::endl;
    return oss.str();
}

} // namespace kaldi
