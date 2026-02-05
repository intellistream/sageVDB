#include "sage_vdb/modality_processors.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
#include <filesystem>

#ifdef OPENCV_ENABLED
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace sage_vdb {

// ===================== Utility helpers =====================
namespace {

inline void l2_normalize(Vector& v) {
    float sum_sq = 0.0f;
    for (float x : v) sum_sq += x * x;
    if (sum_sq > 0) {
        float inv = 1.0f / std::sqrt(sum_sq);
        for (auto& x : v) x *= inv;
    }
}

inline Vector resize_or_pad(const Vector& src, size_t dim) {
    Vector out(dim, 0.0f);
    if (src.size() >= dim) {
        std::copy(src.begin(), src.begin() + dim, out.begin());
    } else {
        std::copy(src.begin(), src.end(), out.begin());
    }
    return out;
}

inline std::string to_lower_copy(const std::string& s) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return std::tolower(c); });
    return t;
}

inline uint32_t hash_str(const std::string& s) {
    return static_cast<uint32_t>(std::hash<std::string>{}(s));
}

inline bool has_extension_magic(const std::vector<uint8_t>& data, const std::vector<std::string>& exts) {
    // Very lightweight magic checks for common formats when possible
    if (data.size() >= 4) {
        // PNG
        if (data.size() >= 8 && data[0]==0x89 && data[1]=='P' && data[2]=='N' && data[3]=='G') return true;
        // JPEG
        if (data[0]==0xFF && data[1]==0xD8) return true;
        // BMP
        if (data[0]=='B' && data[1]=='M') return true;
        // TIFF
        if ((data[0]=='I' && data[1]=='I') || (data[0]=='M' && data[1]=='M')) return true;
        // WAV (RIFF)
        if (data[0]=='R' && data[1]=='I' && data[2]=='F' && data[3]=='F') return true;
        // MP3 ID3v2
        if (data[0]=='I' && data[1]=='D' && data[2]=='3') return true;
    }
    // If unknown, fall back to extension list presence (cannot check without filename) -> best-effort
    return !exts.empty();
}

// Safe reading functions for binary data with endianness conversion
// WAV format uses little-endian byte order
inline uint16_t read_le_uint16(const uint8_t* ptr) {
    uint16_t value;
    std::memcpy(&value, ptr, sizeof(uint16_t));
    // Convert from little-endian to host byte order
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap16(value);
    #else
    return value;
    #endif
}

inline uint32_t read_le_uint32(const uint8_t* ptr) {
    uint32_t value;
    std::memcpy(&value, ptr, sizeof(uint32_t));
    // Convert from little-endian to host byte order
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap32(value);
    #else
    return value;
    #endif
}

inline int16_t read_le_int16(const uint8_t* ptr) {
    int16_t value;
    std::memcpy(&value, ptr, sizeof(int16_t));
    // Convert from little-endian to host byte order
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint16_t uval = static_cast<uint16_t>(value);
    uval = __builtin_bswap16(uval);
    return static_cast<int16_t>(uval);
    #else
    return value;
    #endif
}

inline float read_le_float32(const uint8_t* ptr) {
    uint32_t int_value;
    std::memcpy(&int_value, ptr, sizeof(uint32_t));
    // Convert from little-endian to host byte order
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    int_value = __builtin_bswap32(int_value);
    #endif
    float float_value;
    std::memcpy(&float_value, &int_value, sizeof(float));
    return float_value;
}

} // namespace

// ===================== TextModalityProcessor =====================

TextModalityProcessor::TextModalityProcessor() : config_{} {}
TextModalityProcessor::TextModalityProcessor(const TextConfig& config) : config_(config) {}

std::string TextModalityProcessor::bytes_to_string(const std::vector<uint8_t>& data) const {
    return std::string(reinterpret_cast<const char*>(data.data()), data.size());
}

std::vector<std::string> TextModalityProcessor::tokenize(const std::string& text) const {
    // Simple regex-based tokenizer; if use_bert_tokenization is true, we still use a basic wordpiece-like split
    std::vector<std::string> tokens;
    std::string lower = to_lower_copy(text);
    std::regex re("[a-z0-9_]+", std::regex::icase);
    auto words_begin = std::sregex_iterator(lower.begin(), lower.end(), re);
    auto words_end = std::sregex_iterator();
    for (auto it = words_begin; it != words_end; ++it) {
        std::string tok = it->str();
        if (config_.use_bert_tokenization && tok.size() > 12) {
            // crude wordpiece split for very long tokens
            size_t pos = 0;
            while (pos < tok.size()) {
                size_t len = std::min<size_t>(6, tok.size() - pos);
                tokens.push_back(tok.substr(pos, len));
                pos += len;
            }
        } else {
            tokens.push_back(tok);
        }
    }
    return tokens;
}

Vector TextModalityProcessor::compute_word_embedding_average(const std::vector<std::string>& tokens) const {
    Vector emb(config_.embedding_dim, 0.0f);
    if (tokens.empty()) return emb;
    for (const auto& t : tokens) {
        uint32_t h = hash_str(t);
        size_t idx = h % config_.embedding_dim;
        float weight = 1.0f + 0.1f * static_cast<float>(t.size());
        emb[idx] += weight;
    }
    l2_normalize(emb);
    return emb;
}

Vector TextModalityProcessor::text_to_embedding(const std::string& text) const {
    auto tokens = tokenize(text);
    return compute_word_embedding_average(tokens);
}

Vector TextModalityProcessor::process(const std::vector<uint8_t>& raw_data) {
    if (!validate(raw_data)) {
        throw std::invalid_argument("Invalid TEXT raw data");
    }
    std::string text = bytes_to_string(raw_data);
    return text_to_embedding(text);
}

bool TextModalityProcessor::validate(const std::vector<uint8_t>& raw_data) const {
    if (raw_data.empty()) return false;
    // Check if mostly printable characters
    size_t printable = 0;
    for (auto b : raw_data) {
        if (b=='\n' || b=='\r' || b=='\t' || (b>=32 && b<127)) printable++;
    }
    return printable > raw_data.size() / 2;
}

// ===================== ImageModalityProcessor =====================

ImageModalityProcessor::ImageModalityProcessor() : config_{} {}
ImageModalityProcessor::ImageModalityProcessor(const ImageConfig& config) : config_(config) {}

#ifdef OPENCV_ENABLED
cv::Mat ImageModalityProcessor::bytes_to_mat(const std::vector<uint8_t>& data) const {
    cv::Mat buf(1, static_cast<int>(data.size()), CV_8UC1, const_cast<uint8_t*>(data.data()));
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    return img;
}

cv::Mat ImageModalityProcessor::preprocess_image(const cv::Mat& image) const {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(config_.target_width, config_.target_height));
    if (config_.normalize_pixels) {
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0/255.0);
        return float_img;
    }
    return resized;
}

Vector ImageModalityProcessor::compute_histogram_features(const cv::Mat& image) const {
    // 3-channel color histogram with 32 bins per channel
    const int histSize[] = {32};
    float range[] = {0.0f, config_.normalize_pixels ? 1.0f : 256.0f};
    const float* ranges[] = {range};

    Vector feat;
    feat.reserve(32*3);

    for (int c = 0; c < 3; ++c) {
        cv::Mat hist;
        int channels[] = {c};
        cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
        // normalize hist
        hist /= std::max(1.0, cv::sum(hist)[0]);
        for (int i = 0; i < hist.rows; ++i) feat.push_back(static_cast<float>(hist.at<float>(i)));
    }
    return feat;
}

Vector ImageModalityProcessor::compute_texture_features(const cv::Mat& image) const {
    cv::Mat gray;
    if (image.channels()==3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); else gray = image;
    cv::Mat gradx, grady, lap;
    cv::Sobel(gray, gradx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grady, CV_32F, 0, 1, 3);
    cv::Laplacian(gray, lap, CV_32F);

    auto mean_std = [](const cv::Mat& m){
        cv::Scalar mean, stddev; cv::meanStdDev(m, mean, stddev);
        return std::pair<float,float>(static_cast<float>(mean[0]), static_cast<float>(stddev[0]));
    };

    auto [mx, sx] = mean_std(gradx);
    auto [my, sy] = mean_std(grady);
    auto [ml, sl] = mean_std(lap);

    return Vector{mx, sx, my, sy, ml, sl};
}

Vector ImageModalityProcessor::extract_features(const cv::Mat& image) const {
    Vector h = compute_histogram_features(image);
    Vector t = compute_texture_features(image);
    h.insert(h.end(), t.begin(), t.end());
    return h;
}
#endif

Vector ImageModalityProcessor::process(const std::vector<uint8_t>& raw_data) {
#ifndef OPENCV_ENABLED
    (void)raw_data;
    throw std::runtime_error("Image processing requires OpenCV. Rebuild with -DENABLE_OPENCV=ON");
#else
    if (!validate(raw_data)) {
        throw std::invalid_argument("Invalid IMAGE raw data");
    }
    cv::Mat img = bytes_to_mat(raw_data);
    if (img.empty()) throw std::runtime_error("Failed to decode image bytes");
    cv::Mat pre = preprocess_image(img);
    Vector f = extract_features(pre);
    Vector out = resize_or_pad(f, config_.embedding_dim);
    l2_normalize(out);
    return out;
#endif
}

bool ImageModalityProcessor::validate(const std::vector<uint8_t>& raw_data) const {
    if (raw_data.empty()) return false;
    return is_valid_image_format(raw_data);
}

bool ImageModalityProcessor::is_valid_image_format(const std::vector<uint8_t>& data) const {
    return has_extension_magic(data, config_.supported_formats);
}

// ===================== AudioModalityProcessor =====================

AudioModalityProcessor::AudioModalityProcessor() : config_{} {}
AudioModalityProcessor::AudioModalityProcessor(const AudioConfig& config) : config_(config) {}

// Minimal WAV (PCM 16-bit) parser
std::vector<float> AudioModalityProcessor::bytes_to_audio(const std::vector<uint8_t>& data) const {
    // RIFF header: "RIFF" + size + "WAVE"
    if (data.size() < 44) throw std::runtime_error("Audio bytes too small");
    if (!(data[0]=='R' && data[1]=='I' && data[2]=='F' && data[3]=='F' &&
          data[8]=='W' && data[9]=='A' && data[10]=='V' && data[11]=='E')) {
        throw std::runtime_error("Only WAV (PCM16) is supported without FFmpeg");
    }
    // Find "fmt " and "data" chunks
    size_t pos = 12;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1;
    uint32_t sample_rate = config_.sample_rate;
    uint16_t bits_per_sample = 16;
    const uint8_t* ptr = data.data();
    size_t data_offset = 0; uint32_t data_size = 0;

    while (pos + 8 <= data.size()) {
        uint32_t chunk_id = read_le_uint32(ptr + pos);
        uint32_t chunk_size = read_le_uint32(ptr + pos + 4);
        pos += 8;
        if (pos + chunk_size > data.size()) break;
        if (chunk_id == 0x20746d66) { // 'fmt '
            if (chunk_size < 16) throw std::runtime_error("Invalid fmt chunk");
            audio_format = read_le_uint16(ptr + pos + 0);
            num_channels = read_le_uint16(ptr + pos + 2);
            sample_rate = read_le_uint32(ptr + pos + 4);
            bits_per_sample = read_le_uint16(ptr + pos + 14);
        } else if (chunk_id == 0x61746164) { // 'data'
            data_offset = pos;
            data_size = chunk_size;
            break;
        }
        pos += chunk_size + (chunk_size % 2); // padding
    }
    if (audio_format != 1 || bits_per_sample != 16 || data_size == 0) {
        throw std::runtime_error("Only PCM16 WAV supported");
    }

    size_t samples = data_size / (bits_per_sample/8) / num_channels;
    std::vector<float> out;
    out.reserve(samples);

    for (size_t i = 0; i < samples; ++i) {
        int64_t acc = 0;
        for (uint16_t ch = 0; ch < num_channels; ++ch) {
            size_t sample_offset = data_offset + (i * num_channels + ch) * sizeof(int16_t);
            if (sample_offset + sizeof(int16_t) > data.size()) break;
            int16_t sample = read_le_int16(ptr + sample_offset);
            acc += sample;
        }
        float v = static_cast<float>(acc) / (32768.0f * num_channels);
        out.push_back(v);
    }

    // Resample (very naive) if needed to target sample_rate
    (void)sample_rate; // For simplicity, ignore resampling here
    // Truncate to duration_seconds
    size_t max_samples = static_cast<size_t>(config_.duration_seconds * config_.sample_rate);
    if (out.size() > max_samples) out.resize(max_samples);

    return out;
}

static Vector windowed_stats(const std::vector<float>& x, uint32_t win, uint32_t hop) {
    if (x.empty()) return {};
    Vector feat;
    for (size_t start = 0; start + win <= x.size(); start += hop) {
        float mean = 0.0f, var = 0.0f, zcr = 0.0f, energy = 0.0f;
        for (size_t i = start; i < start + win; ++i) {
            float v = x[i];
            mean += v;
            energy += v * v;
            if (i+1 < start + win) zcr += (x[i] * x[i+1] < 0.0f) ? 1.0f : 0.0f;
        }
        mean /= win;
        for (size_t i = start; i < start + win; ++i) {
            float d = x[i] - mean; var += d * d;
        }
        var /= win; float stdv = std::sqrt(var + 1e-12f);
        zcr /= (win - 1);
        feat.push_back(mean);
        feat.push_back(stdv);
        feat.push_back(energy / win);
        feat.push_back(zcr);
    }
    return feat;
}

Vector AudioModalityProcessor::extract_temporal_features(const std::vector<float>& audio_data) const {
    uint32_t win = std::min<uint32_t>(config_.n_fft, 1024);
    uint32_t hop = std::min<uint32_t>(config_.hop_length, win);
    Vector f = windowed_stats(audio_data, win, hop);
    return f;
}

Vector AudioModalityProcessor::extract_spectral_features(const std::vector<float>& audio_data) const {
    // Compute coarse spectral centroid and bandwidth on windows using naive DFT (small N)
    uint32_t N = std::min<uint32_t>(config_.n_fft, 512);
    uint32_t hop = std::min<uint32_t>(config_.hop_length, N);
    Vector feat;
    if (audio_data.size() < N) return feat;

    const float pi = 3.1415926535f;
    for (size_t start = 0; start + N <= audio_data.size(); start += hop) {
        std::vector<float> mag(N/2+1, 0.0f);
        for (uint32_t k = 0; k <= N/2; ++k) {
            float re = 0.0f, im = 0.0f;
            for (uint32_t n = 0; n < N; ++n) {
                float ang = 2.0f * pi * k * n / N;
                re += audio_data[start+n] * std::cos(ang);
                im -= audio_data[start+n] * std::sin(ang);
            }
            mag[k] = std::sqrt(re*re + im*im) + 1e-6f;
        }
        // centroid and bandwidth
        float num = 0.0f, den = 0.0f;
        for (uint32_t k = 0; k <= N/2; ++k) { num += k * mag[k]; den += mag[k]; }
        float centroid = (den>0? num/den : 0.0f) / (N/2);
        float bw_num = 0.0f;
        for (uint32_t k = 0; k <= N/2; ++k) { float d = k - centroid*(N/2); bw_num += d*d * mag[k]; }
        float bandwidth = den>0 ? std::sqrt(bw_num/den) / (N/2) : 0.0f;
        feat.push_back(centroid);
        feat.push_back(bandwidth);
    }
    return feat;
}

Vector AudioModalityProcessor::extract_mfcc_features(const std::vector<float>& audio_data) const {
    // Placeholder: approximate with aggregated spectral-temporal stats to reach embedding_dim
    Vector t = extract_temporal_features(audio_data);
    Vector s = extract_spectral_features(audio_data);
    t.insert(t.end(), s.begin(), s.end());
    return t;
}

Vector AudioModalityProcessor::process(const std::vector<uint8_t>& raw_data) {
    if (!validate(raw_data)) {
        throw std::invalid_argument("Invalid AUDIO raw data");
    }
    auto wav = bytes_to_audio(raw_data);
    Vector f = extract_mfcc_features(wav);
    Vector out = resize_or_pad(f, config_.embedding_dim);
    l2_normalize(out);
    return out;
}

bool AudioModalityProcessor::validate(const std::vector<uint8_t>& raw_data) const {
    if (raw_data.empty()) return false;
    return is_valid_audio_format(raw_data);
}

bool AudioModalityProcessor::is_valid_audio_format(const std::vector<uint8_t>& data) const {
    return has_extension_magic(data, config_.supported_formats);
}

// ===================== VideoModalityProcessor =====================

VideoModalityProcessor::VideoModalityProcessor() : config_{},
    image_processor_(std::make_shared<ImageModalityProcessor>(ImageModalityProcessor::ImageConfig{})),
    audio_processor_(std::make_shared<AudioModalityProcessor>(AudioModalityProcessor::AudioConfig{})) {}

VideoModalityProcessor::VideoModalityProcessor(const VideoConfig& config)
    : config_(config),
    image_processor_(std::make_shared<ImageModalityProcessor>(ImageModalityProcessor::ImageConfig{})),
      audio_processor_(std::make_shared<AudioModalityProcessor>(AudioModalityProcessor::AudioConfig{})) {}

Vector VideoModalityProcessor::process(const std::vector<uint8_t>& raw_data) {
#ifndef OPENCV_ENABLED
    (void)raw_data;
    throw std::runtime_error("Video processing requires OpenCV (with codec support). Rebuild with -DENABLE_OPENCV=ON");
#else
    if (!validate(raw_data)) {
        throw std::invalid_argument("Invalid VIDEO raw data");
    }
    auto frames = extract_frames(raw_data);
    if (frames.empty()) throw std::runtime_error("Failed to decode video frames");

    // Extract features per frame using image pipeline
    std::vector<Vector> frame_feats;
    frame_feats.reserve(frames.size());
    for (const auto& f : frames) {
        // Reuse image feature extractor directly
        cv::Mat pre;
        cv::resize(f, pre, cv::Size(config_.frame_width, config_.frame_height));
        Vector feat;
        {
            // Re-implement small subset to avoid private access
            // Histogram + texture as in ImageModalityProcessor
            // Convert to float normalized
            cv::Mat float_img; pre.convertTo(float_img, CV_32FC3, 1.0/255.0);
            // Histogram
            const int histSize[] = {32};
            float range[] = {0.0f, 1.0f};
            const float* ranges[] = {range};
            for (int c = 0; c < 3; ++c) {
                cv::Mat hist; int channels[] = {c};
                cv::calcHist(&float_img, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
                hist /= std::max(1.0, cv::sum(hist)[0]);
                for (int i = 0; i < hist.rows; ++i) feat.push_back(static_cast<float>(hist.at<float>(i)));
            }
            // Texture
            cv::Mat gray; cv::cvtColor(pre, gray, cv::COLOR_BGR2GRAY);
            cv::Mat gradx, grady, lap;
            cv::Sobel(gray, gradx, CV_32F, 1, 0, 3);
            cv::Sobel(gray, grady, CV_32F, 0, 1, 3);
            cv::Laplacian(gray, lap, CV_32F);
            auto mean_std = [](const cv::Mat& m){ cv::Scalar mean, stddev; cv::meanStdDev(m, mean, stddev); return std::pair<float,float>(static_cast<float>(mean[0]), static_cast<float>(stddev[0])); };
            auto [mx, sx] = mean_std(gradx);
            auto [my, sy] = mean_std(grady);
            auto [ml, sl] = mean_std(lap);
            feat.push_back(mx); feat.push_back(sx); feat.push_back(my); feat.push_back(sy); feat.push_back(ml); feat.push_back(sl);
        }
        frame_feats.push_back(std::move(feat));
    }

    // Aggregate frame features (average + simple temporal deltas)
    Vector agg(config_.embedding_dim, 0.0f);
    if (!frame_feats.empty()) {
        // Average
        size_t dim = std::min<size_t>(config_.embedding_dim, frame_feats[0].size());
        for (const auto& fv : frame_feats) {
            for (size_t i = 0; i < dim; ++i) agg[i] += fv[i];
        }
        for (size_t i = 0; i < dim; ++i) agg[i] /= static_cast<float>(frame_feats.size());
    }

    l2_normalize(agg);
    return agg;
#endif
}

#ifdef OPENCV_ENABLED
std::vector<cv::Mat> VideoModalityProcessor::extract_frames(const std::vector<uint8_t>& video_data) const {
    // Write to temp file then decode with VideoCapture (simplest cross-platform approach)
    // Use secure temp directory and random filename to avoid race conditions
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
    
    // Generate unique filename using random number generator
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    std::string tmp_filename = "SageVDB_vid_" + std::to_string(dis(gen)) + ".mp4";
    std::filesystem::path tmp_path = tmp_dir / tmp_filename;
    
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to create temporary video file");
        }
        ofs.write(reinterpret_cast<const char*>(video_data.data()), static_cast<std::streamsize>(video_data.size()));
    }

    cv::VideoCapture cap(tmp_path.string());
    if (!cap.isOpened()) {
        // Try without writing extension
        cap.open(tmp_path.string(), cv::CAP_FFMPEG);
    }

    std::vector<cv::Mat> frames;
    if (!cap.isOpened()) {
        // Cleanup and return empty
        std::filesystem::remove(tmp_path);
        return frames;
    }

    double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (frame_count <= 0) frame_count = 100; // fallback
    int total_frames = static_cast<int>(frame_count);

    int maxf = static_cast<int>(config_.max_frames);
    int step = std::max(1, total_frames / std::max(1, maxf));

    cv::Mat frame;
    for (int i = 0; i < total_frames && static_cast<int>(frames.size()) < maxf; i += step) {
        cap.set(cv::CAP_PROP_POS_FRAMES, i);
        if (cap.read(frame) && !frame.empty()) {
            frames.push_back(frame.clone());
        } else {
            break;
        }
    }

    cap.release();
    std::filesystem::remove(tmp_path);
    return frames;
}

Vector VideoModalityProcessor::aggregate_frame_features(const std::vector<Vector>& frame_embeddings) const {
    if (frame_embeddings.empty()) return Vector(config_.embedding_dim, 0.0f);
    size_t dim = std::min<size_t>(config_.embedding_dim, frame_embeddings[0].size());
    Vector agg(dim, 0.0f);
    for (const auto& f : frame_embeddings) {
        for (size_t i = 0; i < dim; ++i) agg[i] += f[i];
    }
    for (size_t i = 0; i < dim; ++i) agg[i] /= static_cast<float>(frame_embeddings.size());
    return resize_or_pad(agg, config_.embedding_dim);
}

Vector VideoModalityProcessor::extract_motion_features(const std::vector<cv::Mat>& frames) const {
    if (frames.size() < 2) return Vector(16, 0.0f);
    // Simple frame-difference statistics
    Vector feat;
    feat.reserve(16);
    double total_mean = 0.0, total_std = 0.0;
    int count = 0;
    for (size_t i = 1; i < frames.size(); ++i) {
        cv::Mat gray1, gray2, diff;
        cv::cvtColor(frames[i-1], gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frames[i], gray2, cv::COLOR_BGR2GRAY);
        cv::absdiff(gray1, gray2, diff);
        cv::Scalar mean, stddev; cv::meanStdDev(diff, mean, stddev);
        total_mean += mean[0]; total_std += stddev[0]; count++;
    }
    if (count>0) {
        feat.push_back(static_cast<float>(total_mean/count));
        feat.push_back(static_cast<float>(total_std/count));
    }
    return feat;
}

Vector VideoModalityProcessor::extract_temporal_features(const std::vector<Vector>& frame_embeddings) const {
    if (frame_embeddings.size() < 2) return {};
    size_t dim = frame_embeddings[0].size();
    Vector deltas(dim, 0.0f);
    for (size_t i = 1; i < frame_embeddings.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            deltas[j] += std::abs(frame_embeddings[i][j] - frame_embeddings[i-1][j]);
        }
    }
    for (auto& v : deltas) v /= static_cast<float>(frame_embeddings.size()-1);
    return deltas;
}
#endif

bool VideoModalityProcessor::validate(const std::vector<uint8_t>& raw_data) const {
    if (raw_data.empty()) return false;
    return is_valid_video_format(raw_data);
}

bool VideoModalityProcessor::is_valid_video_format(const std::vector<uint8_t>& data) const {
    return has_extension_magic(data, config_.supported_formats);
}

// ===================== TabularModalityProcessor =====================

TabularModalityProcessor::TabularModalityProcessor() : config_{} {}
TabularModalityProcessor::TabularModalityProcessor(const TabularConfig& config) : config_(config) {}

static std::vector<std::string> split_line(const std::string& s, char delim) {
    std::vector<std::string> out; out.reserve(16);
    std::string cur; bool in_quotes=false;
    for (size_t i=0;i<s.size();++i){
        char c=s[i];
        if (c=='"') { in_quotes=!in_quotes; continue; }
        if (!in_quotes && c==delim) { out.push_back(cur); cur.clear(); }
        else if (!in_quotes && (c=='\r' || c=='\n')) { continue; }
        else { cur.push_back(c); }
    }
    out.push_back(cur);
    return out;
}

TabularModalityProcessor::TableData TabularModalityProcessor::parse_table_data(const std::vector<uint8_t>& data) const {
    // Currently support CSV/TSV only
    std::string text(reinterpret_cast<const char*>(data.data()), data.size());
    std::stringstream ss(text);
    std::string line;
    TableData tbl;

    char delim = config_.delimiter.empty()? ',': config_.delimiter[0];

    bool header_done = false;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        auto cols = split_line(line, delim);
        if (!header_done) {
            tbl.headers = cols; header_done = true;
        } else {
            tbl.rows.push_back(cols);
        }
        if (tbl.rows.size() >= config_.max_rows) break;
    }

    // Detect column types
    size_t cols = tbl.headers.size();
    tbl.column_types.resize(cols, "numeric");
    for (size_t c = 0; c < cols; ++c) {
        std::vector<std::string> column;
        for (const auto& r : tbl.rows) if (c < r.size()) column.push_back(r[c]);
        tbl.column_types[c] = detect_column_type(column);
    }

    return tbl;
}

std::string TabularModalityProcessor::detect_column_type(const std::vector<std::string>& column) const {
    size_t numeric_cnt=0, text_cnt=0;
    std::regex num_re("^[+-]?(?:[0-9]*\\.[0-9]+|[0-9]+)$");
    for (const auto& s : column) {
        if (std::regex_match(s, num_re)) numeric_cnt++; else text_cnt++;
    }
    if (numeric_cnt >= text_cnt) return "numeric";
    // Heuristic: few unique values -> categorical
    std::unordered_map<std::string,int> freq;
    for (const auto& s : column) freq[s]++;
    if (freq.size() < std::min<size_t>(column.size()/4 + 1, 50)) return "categorical";
    return "text";
}

Vector TabularModalityProcessor::encode_numeric_column(const std::vector<std::string>& column) const {
    std::vector<float> vals;
    vals.reserve(column.size());
    for (const auto& s : column) {
        try { vals.push_back(std::stof(s)); } catch (...) { }
        if (vals.size() >= config_.max_rows) break;
    }
    if (vals.empty()) return {};
    float mean = std::accumulate(vals.begin(), vals.end(), 0.0f) / vals.size();
    float var = 0.0f, minv = vals[0], maxv = vals[0];
    for (auto v : vals) { float d=v-mean; var+=d*d; minv=std::min(minv,v); maxv=std::max(maxv,v);} var/=vals.size();
    float stdv = std::sqrt(var + 1e-12f);
    return Vector{mean, stdv, minv, maxv};
}

Vector TabularModalityProcessor::encode_categorical_column(const std::vector<std::string>& column) const {
    const size_t bins = 32;
    Vector hist(bins, 0.0f);
    for (const auto& s : column) {
        size_t idx = hash_str(s) % bins; hist[idx] += 1.0f;
    }
    // normalize
    float sum = std::accumulate(hist.begin(), hist.end(), 0.0f);
    if (sum > 0) for (auto& x : hist) x /= sum;
    return hist;
}

Vector TabularModalityProcessor::normalize_features(const Vector& features) const {
    Vector out = features;
    l2_normalize(out);
    return out;
}

Vector TabularModalityProcessor::encode_tabular_data(const TableData& table) const {
    Vector feat;
    size_t cols = table.headers.size();
    for (size_t c = 0; c < cols; ++c) {
        std::vector<std::string> column;
        for (const auto& r : table.rows) if (c < r.size()) column.push_back(r[c]);
        Vector f;
        if (table.column_types[c] == "numeric") f = encode_numeric_column(column);
        else if (table.column_types[c] == "categorical") f = encode_categorical_column(column);
        else {
            // text: bag-of-words hashed features
            const size_t bins = 64;
            f.assign(bins, 0.0f);
            std::regex re("[A-Za-z0-9_]+");
            for (const auto& cell : column) {
                for (auto it = std::sregex_iterator(cell.begin(), cell.end(), re);
                     it != std::sregex_iterator(); ++it) {
                    size_t idx = hash_str(it->str()) % bins; f[idx] += 1.0f;
                }
            }
            float sum = std::accumulate(f.begin(), f.end(), 0.0f); if (sum>0) for (auto& x : f) x/=sum;
        }
        feat.insert(feat.end(), f.begin(), f.end());
        if (feat.size() >= config_.embedding_dim) break;
    }
    feat = resize_or_pad(feat, config_.embedding_dim);
    return normalize_features(feat);
}

Vector TabularModalityProcessor::process(const std::vector<uint8_t>& raw_data) {
    if (!validate(raw_data)) throw std::invalid_argument("Invalid TABULAR raw data");
    TableData tbl = parse_table_data(raw_data);
    Vector f = encode_tabular_data(tbl);
    return f;
}

bool TabularModalityProcessor::validate(const std::vector<uint8_t>& raw_data) const {
    if (raw_data.empty()) return false;
    // Check for presence of delimiter in the first few KB
    size_t n = std::min<size_t>(raw_data.size(), 4096);
    for (size_t i = 0; i < n; ++i) if (raw_data[i] == (config_.delimiter.empty()? ',': config_.delimiter[0])) return true;
    // Fallback true to allow JSON or other formats (not implemented)
    return true;
}

// ===================== TimeSeriesModalityProcessor =====================

TimeSeriesModalityProcessor::TimeSeriesModalityProcessor() : config_{} {}
TimeSeriesModalityProcessor::TimeSeriesModalityProcessor(const TimeSeriesConfig& config) : config_(config) {}

std::vector<float> TimeSeriesModalityProcessor::bytes_to_series(const std::vector<uint8_t>& data) const {
    // Try parse as CSV of one column
    std::string text(reinterpret_cast<const char*>(data.data()), data.size());
    std::vector<float> series;
    series.reserve(1024);
    size_t digits = 0, commas = 0;
    for (char c : text) { if (std::isdigit(static_cast<unsigned char>(c))) digits++; if (c==','||c=='\n') commas++; }
    if (digits > 0 && commas > 0) {
        std::stringstream ss(text); std::string tok;
        while (std::getline(ss, tok, ',')) {
            try { series.push_back(std::stof(tok)); } catch (...) {}
            if (series.size() >= config_.max_sequence_length) break;
        }
        if (!series.empty()) return series;
    }
    // Otherwise interpret as binary float32 in little-endian byte order
    // Read each float using safe endianness conversion
    size_t cnt = data.size() / sizeof(float);
    series.reserve(cnt);
    const uint8_t* ptr = data.data();
    for (size_t i = 0; i < cnt && series.size() < config_.max_sequence_length; ++i) {
        series.push_back(read_le_float32(ptr + i * sizeof(float)));
    }
    return series;
}

std::vector<float> TimeSeriesModalityProcessor::normalize_series(const std::vector<float>& s) const {
    if (!config_.normalize_series) return s;
    std::vector<float> out = s;
    float mean = 0.0f; for (auto v : out) mean += v; mean /= std::max<size_t>(1, out.size());
    float var = 0.0f; for (auto v : out) { float d=v-mean; var += d*d; }
    var /= std::max<size_t>(1, out.size()); float stdv = std::sqrt(var + 1e-12f);
    if (stdv > 0) for (auto& v : out) v = (v - mean) / stdv;
    return out;
}

Vector TimeSeriesModalityProcessor::extract_statistical_features(const std::vector<float>& series) const {
    if (series.empty()) return {};
    float mean = 0.0f, minv = series[0], maxv = series[0];
    for (auto v : series) { mean += v; minv = std::min(minv, v); maxv = std::max(maxv, v);} mean /= series.size();
    float var=0.0f; for (auto v : series) { float d=v-mean; var+=d*d; } var/=series.size(); float stdv = std::sqrt(var+1e-12f);
    return Vector{mean, stdv, minv, maxv};
}

Vector TimeSeriesModalityProcessor::extract_frequency_features(const std::vector<float>& series) const {
    // Coarse FFT magnitude at a few frequencies using naive DFT
    uint32_t N = std::min<uint32_t>(static_cast<uint32_t>(series.size()), 256);
    if (N < 16) return {};
    const float pi = 3.1415926535f;
    Vector mags;
    for (uint32_t k : {1u,2u,3u,4u,5u,6u,8u,12u,16u,24u,32u,48u,64u}) {
        if (k >= N) break;
        float re=0.0f, im=0.0f;
        for (uint32_t n=0; n<N; ++n) {
            float ang = 2.0f*pi*k*n/N; re += series[n]*std::cos(ang); im -= series[n]*std::sin(ang);
        }
        mags.push_back(std::sqrt(re*re+im*im));
    }
    l2_normalize(mags);
    return mags;
}

Vector TimeSeriesModalityProcessor::extract_trend_features(const std::vector<float>& series) const {
    if (!config_.extract_trend || series.size()<2) return {};
    // Simple linear regression slope
    double sumx=0,sumy=0,sumxy=0,sumxx=0; size_t n=series.size();
    for (size_t i=0;i<n;++i){ sumx+=i; sumy+=series[i]; sumxy+=i*series[i]; sumxx+=i*i; }
    double denom = n*sumxx - sumx*sumx; double slope=0.0; if (denom!=0) slope = (n*sumxy - sumx*sumy)/denom;
    return Vector{static_cast<float>(slope)};
}

Vector TimeSeriesModalityProcessor::extract_window_features(const std::vector<float>& series) const {
    uint32_t win = std::max<uint32_t>(1, config_.window_size);
    uint32_t stride = std::max<uint32_t>(1, config_.stride);
    Vector feat;
    for (size_t start = 0; start + win <= series.size(); start += stride) {
        float mean=0; for (size_t i=0;i<win;++i) mean+=series[start+i]; mean/=win;
        feat.push_back(mean);
    }
    l2_normalize(feat);
    return feat;
}

Vector TimeSeriesModalityProcessor::process(const std::vector<uint8_t>& raw_data) {
    if (!validate(raw_data)) throw std::invalid_argument("Invalid TIME_SERIES raw data");
    auto s = bytes_to_series(raw_data);
    s = normalize_series(s);
    Vector f;
    auto f1 = extract_statistical_features(s); f.insert(f.end(), f1.begin(), f1.end());
    if (config_.extract_seasonality) { auto f2 = extract_frequency_features(s); f.insert(f.end(), f2.begin(), f2.end()); }
    if (config_.extract_trend) { auto f3 = extract_trend_features(s); f.insert(f.end(), f3.begin(), f3.end()); }
    auto f4 = extract_window_features(s); f.insert(f.end(), f4.begin(), f4.end());
    f = resize_or_pad(f, config_.embedding_dim); l2_normalize(f);
    return f;
}

bool TimeSeriesModalityProcessor::validate(const std::vector<uint8_t>& raw_data) const {
    return !raw_data.empty();
}

// ===================== ModalityProcessorFactory =====================

std::unordered_map<std::string, std::function<std::shared_ptr<ModalityProcessor>()>> ModalityProcessorFactory::custom_processors_;

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_text_processor(TextModalityProcessor::TextConfig config) {
    return std::make_shared<TextModalityProcessor>(config);
}

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_image_processor(ImageModalityProcessor::ImageConfig config) {
    return std::make_shared<ImageModalityProcessor>(config);
}

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_audio_processor(AudioModalityProcessor::AudioConfig config) {
    return std::make_shared<AudioModalityProcessor>(config);
}

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_video_processor(VideoModalityProcessor::VideoConfig config) {
    return std::make_shared<VideoModalityProcessor>(config);
}

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_tabular_processor(TabularModalityProcessor::TabularConfig config) {
    return std::make_shared<TabularModalityProcessor>(config);
}

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_time_series_processor(TimeSeriesModalityProcessor::TimeSeriesConfig config) {
    return std::make_shared<TimeSeriesModalityProcessor>(config);
}

std::unordered_map<ModalityType, std::shared_ptr<ModalityProcessor>> ModalityProcessorFactory::create_standard_processors() {
    std::unordered_map<ModalityType, std::shared_ptr<ModalityProcessor>> m;
    m[ModalityType::TEXT] = create_text_processor();
    m[ModalityType::IMAGE] = create_image_processor();
    m[ModalityType::AUDIO] = create_audio_processor();
    m[ModalityType::VIDEO] = create_video_processor();
    m[ModalityType::TABULAR] = create_tabular_processor();
    m[ModalityType::TIME_SERIES] = create_time_series_processor();
    return m;
}

void ModalityProcessorFactory::register_custom_processor(const std::string& name, std::function<std::shared_ptr<ModalityProcessor>()> factory_func) {
    custom_processors_[name] = std::move(factory_func);
}

std::shared_ptr<ModalityProcessor> ModalityProcessorFactory::create_custom_processor(const std::string& name) {
    auto it = custom_processors_.find(name);
    if (it == custom_processors_.end()) return nullptr;
    return it->second();
}

} // namespace sage_vdb
