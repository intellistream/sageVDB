#pragma once

#include "multimodal_fusion.h"
#ifdef OPENCV_ENABLED
#include <opencv2/opencv.hpp>  // Requires OpenCV library
#endif
#include <string>
#include <regex>

namespace sage_vdb {

// ========== 文本模态处理器 ==========
class TextModalityProcessor : public ModalityProcessor {
public:
    struct TextConfig {
        uint32_t max_sequence_length = 512;
        uint32_t embedding_dim = 768;
        std::string model_path = "";  // 文本嵌入模型路径
        bool use_bert_tokenization = true;
    };
    
    TextModalityProcessor();
    explicit TextModalityProcessor(const TextConfig& config);
    
    Vector process(const std::vector<uint8_t>& raw_data) override;
    bool validate(const std::vector<uint8_t>& raw_data) const override;
    ModalityType get_type() const override { return ModalityType::TEXT; }

private:
    TextConfig config_;
    
    std::string bytes_to_string(const std::vector<uint8_t>& data) const;
    Vector text_to_embedding(const std::string& text) const;
    std::vector<std::string> tokenize(const std::string& text) const;
    Vector compute_word_embedding_average(const std::vector<std::string>& tokens) const;
};

// ========== 图像模态处理器 ==========
class ImageModalityProcessor : public ModalityProcessor {
public:
    struct ImageConfig {
        uint32_t target_width = 224;
        uint32_t target_height = 224;
        uint32_t embedding_dim = 2048;
        std::string model_path = "";  // 图像特征提取模型路径
        bool normalize_pixels = true;
        std::vector<std::string> supported_formats = {"jpg", "jpeg", "png", "bmp", "tiff"};
    };
    
    ImageModalityProcessor();
    explicit ImageModalityProcessor(const ImageConfig& config);
    
    Vector process(const std::vector<uint8_t>& raw_data) override;
    bool validate(const std::vector<uint8_t>& raw_data) const override;
    ModalityType get_type() const override { return ModalityType::IMAGE; }

private:
    ImageConfig config_;
#ifdef OPENCV_ENABLED
    cv::Mat bytes_to_mat(const std::vector<uint8_t>& data) const;
    cv::Mat preprocess_image(const cv::Mat& image) const;
    Vector extract_features(const cv::Mat& image) const;
    Vector compute_histogram_features(const cv::Mat& image) const;
    Vector compute_texture_features(const cv::Mat& image) const;
#endif
    bool is_valid_image_format(const std::vector<uint8_t>& data) const;
};

// ========== 音频模态处理器 ==========
class AudioModalityProcessor : public ModalityProcessor {
public:
    struct AudioConfig {
        uint32_t sample_rate = 16000;
        uint32_t n_mfcc = 13;           // MFCC特征数量
        uint32_t n_fft = 2048;          // FFT窗口大小
        uint32_t hop_length = 512;      // 帧移
        uint32_t embedding_dim = 512;
        float duration_seconds = 10.0f;  // 最大音频长度
        std::vector<std::string> supported_formats = {"wav", "mp3", "flac", "m4a"};
    };
    
    AudioModalityProcessor();
    explicit AudioModalityProcessor(const AudioConfig& config);
    
    Vector process(const std::vector<uint8_t>& raw_data) override;
    bool validate(const std::vector<uint8_t>& raw_data) const override;
    ModalityType get_type() const override { return ModalityType::AUDIO; }

private:
    AudioConfig config_;
    
    std::vector<float> bytes_to_audio(const std::vector<uint8_t>& data) const;
    Vector extract_mfcc_features(const std::vector<float>& audio_data) const;
    Vector extract_spectral_features(const std::vector<float>& audio_data) const;
    Vector extract_temporal_features(const std::vector<float>& audio_data) const;
    std::vector<float> apply_window(const std::vector<float>& signal, uint32_t start, uint32_t length) const;
    bool is_valid_audio_format(const std::vector<uint8_t>& data) const;
};

// ========== 视频模态处理器 ==========
class VideoModalityProcessor : public ModalityProcessor {
public:
    struct VideoConfig {
        uint32_t max_frames = 32;
        uint32_t frame_width = 224;
        uint32_t frame_height = 224;
        uint32_t embedding_dim = 1024;
        float fps_sampling = 1.0f;      // 每秒采样帧数
        bool extract_audio = true;      // 是否同时提取音频特征
        std::vector<std::string> supported_formats = {"mp4", "avi", "mov", "mkv"};
    };
    
    VideoModalityProcessor();
    explicit VideoModalityProcessor(const VideoConfig& config);
    
    Vector process(const std::vector<uint8_t>& raw_data) override;
    bool validate(const std::vector<uint8_t>& raw_data) const override;
    ModalityType get_type() const override { return ModalityType::VIDEO; }

private:
    VideoConfig config_;
    std::shared_ptr<ImageModalityProcessor> image_processor_;
    std::shared_ptr<AudioModalityProcessor> audio_processor_;
    
#ifdef OPENCV_ENABLED
    std::vector<cv::Mat> extract_frames(const std::vector<uint8_t>& video_data) const;
    Vector aggregate_frame_features(const std::vector<Vector>& frame_embeddings) const;
    Vector extract_motion_features(const std::vector<cv::Mat>& frames) const;
    Vector extract_temporal_features(const std::vector<Vector>& frame_embeddings) const;
#endif
    bool is_valid_video_format(const std::vector<uint8_t>& data) const;
};

// ========== 表格数据模态处理器 ==========
class TabularModalityProcessor : public ModalityProcessor {
public:
    struct TabularConfig {
        uint32_t max_columns = 100;
        uint32_t max_rows = 1000;
        uint32_t embedding_dim = 256;
        bool normalize_features = true;
        bool handle_missing_values = true;
        std::string delimiter = ",";
        std::vector<std::string> supported_formats = {"csv", "tsv", "json"};
    };
    
    TabularModalityProcessor();
    explicit TabularModalityProcessor(const TabularConfig& config);
    
    Vector process(const std::vector<uint8_t>& raw_data) override;
    bool validate(const std::vector<uint8_t>& raw_data) const override;
    ModalityType get_type() const override { return ModalityType::TABULAR; }

private:
    TabularConfig config_;
    
    struct TableData {
        std::vector<std::vector<std::string>> rows;
        std::vector<std::string> headers;
        std::vector<std::string> column_types;  // "numeric", "categorical", "text"
    };
    
    TableData parse_table_data(const std::vector<uint8_t>& data) const;
    Vector encode_tabular_data(const TableData& table) const;
    Vector encode_categorical_column(const std::vector<std::string>& column) const;
    Vector encode_numeric_column(const std::vector<std::string>& column) const;
    Vector normalize_features(const Vector& features) const;
    std::string detect_column_type(const std::vector<std::string>& column) const;
};

// ========== 时间序列模态处理器 ==========
class TimeSeriesModalityProcessor : public ModalityProcessor {
public:
    struct TimeSeriesConfig {
        uint32_t max_sequence_length = 1000;
        uint32_t embedding_dim = 128;
        uint32_t window_size = 50;
        uint32_t stride = 10;
        bool normalize_series = true;
        bool extract_trend = true;
        bool extract_seasonality = true;
    };
    
    TimeSeriesModalityProcessor();
    explicit TimeSeriesModalityProcessor(const TimeSeriesConfig& config);
    
    Vector process(const std::vector<uint8_t>& raw_data) override;
    bool validate(const std::vector<uint8_t>& raw_data) const override;
    ModalityType get_type() const override { return ModalityType::TIME_SERIES; }

private:
    TimeSeriesConfig config_;
    
    std::vector<float> bytes_to_series(const std::vector<uint8_t>& data) const;
    Vector extract_statistical_features(const std::vector<float>& series) const;
    Vector extract_frequency_features(const std::vector<float>& series) const;
    Vector extract_trend_features(const std::vector<float>& series) const;
    Vector extract_window_features(const std::vector<float>& series) const;
    std::vector<float> normalize_series(const std::vector<float>& series) const;
};

// ========== 模态处理器工厂 ==========
class ModalityProcessorFactory {
public:
    // 创建标准处理器
    static std::shared_ptr<ModalityProcessor> create_text_processor(
        TextModalityProcessor::TextConfig config = {});
    
    static std::shared_ptr<ModalityProcessor> create_image_processor(
        ImageModalityProcessor::ImageConfig config = {});
    
    static std::shared_ptr<ModalityProcessor> create_audio_processor(
        AudioModalityProcessor::AudioConfig config = {});
    
    static std::shared_ptr<ModalityProcessor> create_video_processor(
        VideoModalityProcessor::VideoConfig config = {});
    
    static std::shared_ptr<ModalityProcessor> create_tabular_processor(
        TabularModalityProcessor::TabularConfig config = {});
    
    static std::shared_ptr<ModalityProcessor> create_time_series_processor(
        TimeSeriesModalityProcessor::TimeSeriesConfig config = {});
    
    // 批量创建处理器
    static std::unordered_map<ModalityType, std::shared_ptr<ModalityProcessor>> 
    create_standard_processors();
    
    // 注册自定义处理器
    static void register_custom_processor(
        const std::string& name,
        std::function<std::shared_ptr<ModalityProcessor>()> factory_func);
    
    static std::shared_ptr<ModalityProcessor> create_custom_processor(
        const std::string& name);

private:
    static std::unordered_map<std::string, std::function<std::shared_ptr<ModalityProcessor>()>> custom_processors_;
};

} // namespace sage_vdb