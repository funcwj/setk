// perform-snmf.cc
// wujian@2018


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

#include "include/stft.h"
#include "include/snmf.h"



using namespace kaldi;

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Do sparse NMF and create W/H matrix for inputs.\n"
            "\n"
            "Usage:  perform-snmf [options...] <wav-rspecifier> <basis-wspecifier> (atom-wspecifier)\n"
            "   or:  perform-snmf [options...] <wav-rxfilename> <basis-rxfilename> (atom-wxfilename)\n";

        ParseOptions po(usage);
        ShortTimeFTOptions stft_options;
        SparseNMFOptions snmf_options;

        int32 num_atoms = 64, rand_seed = 777;
        bool wx_binary = true;

        po.Register("num-atoms", &num_atoms, "Number of columns defined for W matrix(dictionary size)");
        po.Register("random-seed", &rand_seed, "Seed for random number generator");
        po.Register("binary", &wx_binary, "Write in binary mode (only relevant if output is a wxfilename)");

        stft_options.Register(&po);
        snmf_options.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() < 2 || po.NumArgs() > 3) {
            po.PrintUsage();
            exit(1);
        }

        std::srand(rand_seed);
        // log cause negative elements
        stft_options.apply_log = false;
        ShortTimeFTComputer stft_computer(stft_options);
        SparseNMF snmf(snmf_options);

        std::string wave_in = po.GetArg(1), basis_out = po.GetArg(2), atom_out = "";

        if (po.NumArgs() == 3)
            atom_out = po.GetArg(3);

        bool wav_is_rspecifier = (ClassifyRspecifier(wave_in, NULL, NULL) != kNoRspecifier),
            basis_is_wspecifier = (ClassifyWspecifier(basis_out, NULL, NULL, NULL) != kNoWspecifier);
        
        if (wav_is_rspecifier != basis_is_wspecifier)
            KALDI_ERR << "Cannot mix archives with regular files";

        int32 num_frames, num_bins;

        if (wav_is_rspecifier) {

            SequentialTableReader<WaveHolder> wave_reader(wave_in);
            BaseFloatMatrixWriter basis_writer, atom_writer;

            if (!basis_writer.Open(basis_out))
                KALDI_ERR << "Could not initialize output with wspecifier " << basis_out;
            if (atom_out != "" && !basis_writer.Open(atom_out))
                KALDI_ERR << "Could not initialize output with wspecifier " << atom_out;

            int32 num_utts = 0;
            for (; !wave_reader.Done(); wave_reader.Next()) {
                std::string utt_key = wave_reader.Key();
                const WaveData &wave_data = wave_reader.Value();
                
                // compute spectrum (T x F)
                Matrix<BaseFloat> feature;
                stft_computer.Compute(wave_data.Data(), NULL, &feature, NULL);
                num_frames = feature.NumRows(), num_bins = feature.NumCols();
                // (F x T)
                feature.Transpose();
                Matrix<BaseFloat> W(num_bins, num_atoms), H(num_atoms, num_frames);
                W.SetRandUniform(), H.SetRandUniform();
                snmf.DoSparseNMF(feature, &W, &H);
                
                basis_writer.Write(utt_key, W);
                if (atom_out != "")
                    atom_writer.Write(utt_key, H);
                
                num_utts += 1;
                if (num_utts % 100 == 0)
                    KALDI_LOG << "Processed " << num_utts << " utterances";
                KALDI_VLOG(2) << "Perform sparse NMF for utterance " << utt_key;
            }

            KALDI_LOG << "Done " << num_utts << " utterances";
            return num_utts == 0 ? 1: 0;
        } else {
            bool binary;
            Input ki(wave_in, &binary);
            WaveData wave_input;
            wave_input.Read(ki.Stream());

            Matrix<BaseFloat> feature;
            stft_computer.Compute(wave_input.Data(), NULL, &feature, NULL);
            num_frames = feature.NumRows(), num_bins = feature.NumCols();

            feature.Transpose();
            Matrix<BaseFloat> W(num_bins, num_atoms), H(num_atoms, num_frames);
            W.SetRandUniform(), H.SetRandUniform();
            snmf.DoSparseNMF(feature, &W, &H);

            WriteKaldiObject(W, basis_out, wx_binary);
            if (atom_out != "")
                WriteKaldiObject(H, atom_out, wx_binary);

            KALDI_LOG << "Done processed " << wave_in << " done";

        }

        } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}