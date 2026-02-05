#include "sage_vdb/anns/flat_gpu_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <cctype>
#include <numeric>
#include <queue>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <cstring>

#ifdef ENABLE_LIBAMM
#include "LibAMM.h"
#endif

#include "sage_vdb/anns/flat_gpu/cuda_helpers.h"

namespace sage_vdb {
namespace anns {

namespace {
REGISTER_ANNS_ALGORITHM(FlatGPUANNSFactory);

constexpr uint32_t kFlatGPUSerializationMagic = 0x53414745; // "SAGE"
constexpr uint32_t kFlatGPUSerializationVersion = 1;

struct FlatGPUHostConfig {
    size_t capacity = 0;
    size_t mem_buffer_size = 0;
    size_t dco_batch_size = 0;
    size_t sketch_size = 0;
    std::string amm_algo = "mm";
    bool libamm_use_cuda = false;
};

#ifdef ENABLE_LIBAMM
static inline at::Tensor tensor_from_span(const float* data,
                                          size_t rows,
                                          size_t cols,
                                          at::Device device) {
    if (rows == 0 || cols == 0) {
        return at::empty({static_cast<long>(rows), static_cast<long>(cols)},
                         at::TensorOptions().dtype(at::kFloat).device(device));
    }

    at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat).device(device);
    at::Tensor owned = at::empty({static_cast<long>(rows), static_cast<long>(cols)}, opts);
    std::memcpy(owned.data_ptr<float>(), data, rows * cols * sizeof(float));
    return owned;
}
#endif

FlatGPUHostConfig extract_host_config(const AlgorithmParams& params, size_t dataset_size) {
    FlatGPUHostConfig cfg;
    cfg.capacity = params.get<size_t>("capacity", dataset_size);
    cfg.capacity = std::max(cfg.capacity, dataset_size);

    cfg.mem_buffer_size = params.get<size_t>("memBufferSize", cfg.capacity);
    cfg.mem_buffer_size = std::max(cfg.mem_buffer_size, cfg.capacity);

    cfg.sketch_size = params.get<size_t>("sketchSize", 10);
    if (cfg.sketch_size == 0) {
        cfg.sketch_size = 10;
    }

    cfg.dco_batch_size = params.get<size_t>("DCOBatchSize", cfg.mem_buffer_size);
    if (cfg.dco_batch_size == 0) {
        cfg.dco_batch_size = cfg.mem_buffer_size;
    }

    cfg.amm_algo = params.get<std::string>("ammAlgo", "mm");
    std::transform(cfg.amm_algo.begin(), cfg.amm_algo.end(), cfg.amm_algo.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (cfg.amm_algo != "mm" && cfg.amm_algo != "crs" && cfg.amm_algo != "smp-pca") {
        cfg.amm_algo = "mm";
    }

    cfg.libamm_use_cuda = params.get<bool>("libammUseCuda", false);

    return cfg;
}
}

class FlatGPUANNS::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    void reset() {
        data_.clear();
        ids_.clear();
        id_to_index_.clear();
        capacity_ = 0;
        memory_read_cnt_total_ = 0;
        memory_write_cnt_total_ = 0;
        memory_read_cnt_miss_ = 0;
        memory_write_cnt_miss_ = 0;
        device_buffers_.count = 0;
        device_buffers_.capacity = 0;
        device_buffers_.vectors.reset();
        device_buffers_.ids.reset();
        device_buffers_.norms.reset();
    device_buffers_.dimension = 0;
        query_scratch_ = flat_gpu::QueryScratch{};
        cuda_backend_.reset();
        cuda_enabled_ = false;
        cuda_dirty_ = false;
        cuda_device_id_ = -1;
        last_upload_stats_ = flat_gpu::DeviceStats{};
        last_query_stats_ = flat_gpu::DeviceStats{};
        mem_buffer_size_ = 0;
        dco_batch_size_ = 0;
        sketch_size_ = 0;
        amm_algo_ = "mm";
    libamm_use_cuda_ = false;
    }

    void init(uint32_t dimension, size_t reserve_count) {
        dimension_ = dimension;
        capacity_ = reserve_count;
        data_.reserve(static_cast<size_t>(dimension) * reserve_count);
        ids_.reserve(reserve_count);
    }

    size_t size() const { return ids_.size(); }

    size_t capacity() const { return capacity_; }

    void append(VectorId id, const Vector& vec) {
        if (vec.size() != dimension_) {
            throw std::runtime_error("FlatGPUANNS: vector dimension mismatch on append");
        }
        if (id_to_index_.count(id) > 0) {
            throw std::runtime_error("FlatGPUANNS: duplicate vector id");
        }
        ensure_capacity(size() + 1);
        size_t index = ids_.size();
        ids_.push_back(id);
        id_to_index_[id] = ids_.size() - 1;
        memory_write_cnt_total_++;
        size_t offset = index * dimension_;
        if (data_.size() < offset + dimension_) {
            data_.resize(offset + dimension_);
        }
        std::copy(vec.begin(), vec.end(), data_.begin() + offset);
    }

    void append_bulk(const std::vector<VectorEntry>& entries) {
        if (entries.empty()) {
            return;
        }
        const size_t old_size = size();
        const size_t total_needed = old_size + entries.size();
        ensure_capacity(total_needed);
        ids_.resize(total_needed);
        data_.resize(static_cast<size_t>(dimension_) * total_needed);

        size_t batch = mem_buffer_size_ == 0 ? entries.size() : mem_buffer_size_;
        batch = std::max<size_t>(batch, 1);

        size_t processed = 0;
        size_t dest_index = old_size;
        size_t dest_offset = static_cast<size_t>(old_size) * dimension_;

        while (processed < entries.size()) {
            size_t chunk = std::min(batch, entries.size() - processed);
            for (size_t i = 0; i < chunk; ++i) {
                const auto& entry = entries[processed + i];
                if (entry.second.size() != dimension_) {
                    throw std::runtime_error("FlatGPUANNS: vector dimension mismatch on bulk append");
                }
                if (id_to_index_.count(entry.first) > 0) {
                    throw std::runtime_error("FlatGPUANNS: duplicate vector id in bulk append");
                }
                ids_[dest_index] = entry.first;
                id_to_index_[entry.first] = dest_index;
                std::copy(entry.second.begin(), entry.second.end(), data_.begin() + dest_offset);
                memory_write_cnt_total_++;
                ++dest_index;
                dest_offset += dimension_;
            }
            processed += chunk;
        }
    }

    bool remove(VectorId id) {
        auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) {
            memory_write_cnt_miss_++;
            return false;
        }
        memory_write_cnt_total_++;
        size_t idx = it->second;
        size_t last_idx = ids_.size() - 1;
        if (idx != last_idx) {
            std::copy_n(row_ptr(last_idx), dimension_, row_ptr(idx));
            ids_[idx] = ids_[last_idx];
            id_to_index_[ids_[idx]] = idx;
        }
        ids_.pop_back();
        id_to_index_.erase(it);
        data_.resize(static_cast<size_t>(ids_.size()) * dimension_);
        return true;
    }

    const float* row_ptr(size_t index) const {
        memory_read_cnt_total_++;
        if (index >= ids_.size()) {
            memory_read_cnt_miss_++;
            return nullptr;
        }
        return data_.data() + index * dimension_;
    }

    float* row_ptr(size_t index) {
        memory_read_cnt_total_++;
        if (index >= ids_.size()) {
            memory_read_cnt_miss_++;
            return nullptr;
        }
        return data_.data() + index * dimension_;
    }

    size_t index_of(VectorId id) const {
        auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) {
            throw std::runtime_error("FlatGPUANNS: unknown vector id");
        }
        return it->second;
    }

    size_t dimension() const { return dimension_; }

    size_t memory_usage_bytes() const {
        return data_.size() * sizeof(float) + ids_.size() * sizeof(VectorId) +
               id_to_index_.bucket_count() * sizeof(void*);
    }

    const std::vector<VectorId>& ids() const { return ids_; }

    const std::vector<float>& raw_data() const { return data_; }

    uint64_t memory_read_cnt_total() const { return memory_read_cnt_total_; }
    uint64_t memory_read_cnt_miss() const { return memory_read_cnt_miss_; }
    uint64_t memory_write_cnt_total() const { return memory_write_cnt_total_; }
    uint64_t memory_write_cnt_miss() const { return memory_write_cnt_miss_; }

    void configure_cuda(uint32_t dimension, const AlgorithmParams& params) {
        device_buffers_.dimension = dimension;
        cuda_enabled_ = params.get<bool>("enableGPU", true);
        if (!cuda_enabled_) {
            cuda_backend_.reset();
            cuda_dirty_ = false;
            return;
        }

        int requested_device = params.get<int64_t>("cudaDevice", -1);
        if (!flat_gpu::CUDABackend::is_available()) {
            cuda_enabled_ = false;
            cuda_backend_.reset();
            return;
        }

        cuda_backend_ = flat_gpu::CUDABackend::create(static_cast<int>(requested_device));
        if (!cuda_backend_) {
            cuda_enabled_ = false;
            return;
        }
        cuda_device_id_ = requested_device;
        cuda_dirty_ = true;
    }

    bool using_cuda() const {
        return cuda_enabled_ && cuda_backend_ != nullptr;
    }

    void mark_cuda_dirty() {
        if (using_cuda()) {
            cuda_dirty_ = true;
        }
    }

    void sync_cuda() {
        if (!using_cuda() || !cuda_dirty_) {
            return;
        }
        flat_gpu::DeviceStats stats;
        cuda_backend_->upload(device_buffers_, data_, ids_, stats);
        last_upload_stats_ = stats;
        cuda_dirty_ = false;
    }

    flat_gpu::DeviceBuffers& device_buffers() { return device_buffers_; }
    flat_gpu::QueryScratch& query_scratch() { return query_scratch_; }
    flat_gpu::CUDABackend* cuda_backend() { return cuda_backend_.get(); }

    void set_last_query_stats(const flat_gpu::DeviceStats& stats) {
        last_query_stats_ = stats;
    }

    const flat_gpu::DeviceStats& last_query_stats() const { return last_query_stats_; }
    const flat_gpu::DeviceStats& last_upload_stats() const { return last_upload_stats_; }
    bool cuda_dirty() const { return cuda_dirty_; }

    void configure_host(size_t mem_buffer_size,
                        size_t dco_batch_size,
                        size_t sketch_size,
                        std::string amm_algo,
                        bool libamm_use_cuda) {
        mem_buffer_size_ = mem_buffer_size;
        dco_batch_size_ = dco_batch_size;
        sketch_size_ = sketch_size;
        amm_algo_ = std::move(amm_algo);
        libamm_use_cuda_ = libamm_use_cuda;
    }

    size_t mem_buffer_size() const { return mem_buffer_size_; }
    size_t dco_batch_size() const { return dco_batch_size_; }
    size_t sketch_size() const { return sketch_size_; }
    const std::string& amm_algo() const { return amm_algo_; }
    bool libamm_use_cuda() const { return libamm_use_cuda_; }

private:
    void ensure_capacity(size_t desired) {
        if (desired <= capacity_) {
            return;
        }
        size_t growth_step = mem_buffer_size_ > 0 ? mem_buffer_size_ : (capacity_ == 0 ? desired : capacity_);
        if (capacity_ == 0) {
            capacity_ = std::max(desired, growth_step);
        } else {
            capacity_ = std::max(desired, capacity_ + growth_step);
        }
        data_.reserve(static_cast<size_t>(dimension_) * capacity_);
        ids_.reserve(capacity_);
    }

    uint32_t dimension_ = 0;
    size_t capacity_ = 0;
    std::vector<float> data_;
    std::vector<VectorId> ids_;
    std::unordered_map<VectorId, size_t> id_to_index_;

    mutable uint64_t memory_read_cnt_total_ = 0;
    mutable uint64_t memory_read_cnt_miss_ = 0;
    mutable uint64_t memory_write_cnt_total_ = 0;
    mutable uint64_t memory_write_cnt_miss_ = 0;

    flat_gpu::DeviceBuffers device_buffers_;
    flat_gpu::QueryScratch query_scratch_;
    std::unique_ptr<flat_gpu::CUDABackend> cuda_backend_;
    bool cuda_enabled_ = false;
    bool cuda_dirty_ = false;
    int cuda_device_id_ = -1;
    flat_gpu::DeviceStats last_upload_stats_;
    flat_gpu::DeviceStats last_query_stats_;
    size_t mem_buffer_size_ = 0;
    size_t dco_batch_size_ = 0;
    size_t sketch_size_ = 0;
    std::string amm_algo_ = "mm";
    bool libamm_use_cuda_ = false;
};

FlatGPUANNS::FlatGPUANNS()
    : impl_(std::make_unique<Impl>()),
      built_(false),
      metric_(DistanceMetric::L2),
      dimension_(0) {
    metrics_.reset();
}

FlatGPUANNS::~FlatGPUANNS() = default;

std::string FlatGPUANNS::version() const {
    return "1.0.0";
}

std::string FlatGPUANNS::description() const {
    return "Flat brute-force ANN with optional GPU acceleration hooks";
}

std::vector<DistanceMetric> FlatGPUANNS::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

bool FlatGPUANNS::supports_distance(DistanceMetric metric) const {
    auto supported = supported_distances();
    return std::find(supported.begin(), supported.end(), metric) != supported.end();
}

static float compute_distance(DistanceMetric metric,
                              const float* a,
                              const Vector& b,
                              uint32_t dim,
                              float norm_a_cached = -1.0f) {
    switch (metric) {
        case DistanceMetric::L2: {
            float dist = 0.0f;
            for (uint32_t i = 0; i < dim; ++i) {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }
            return std::sqrt(dist);
        }
        case DistanceMetric::INNER_PRODUCT: {
            float result = 0.0f;
            for (uint32_t i = 0; i < dim; ++i) {
                result += a[i] * b[i];
            }
            return result; // higher is better
        }
        case DistanceMetric::COSINE: {
            float dot = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;
            if (norm_a_cached >= 0.0f) {
                norm_a = norm_a_cached;
            }
            for (uint32_t i = 0; i < dim; ++i) {
                dot += a[i] * b[i];
                if (norm_a_cached < 0.0f) {
                    norm_a += a[i] * a[i];
                }
                norm_b += b[i] * b[i];
            }
            if (norm_a_cached < 0.0f) {
                norm_a = std::sqrt(norm_a);
            }
            norm_b = std::sqrt(norm_b);
            if (norm_a == 0.0f || norm_b == 0.0f) {
                return 1.0f; // maximal distance when zero vector present
            }
            return 1.0f - (dot / (norm_a * norm_b));
        }
    }
    return 0.0f;
}

void FlatGPUANNS::fit(const std::vector<VectorEntry>& dataset,
                      const AlgorithmParams& params) {
    metrics_.reset();
    auto start = std::chrono::high_resolution_clock::now();

    build_params_ = params;
    metric_ = static_cast<DistanceMetric>(
        params.get<int>("metric", static_cast<int>(DistanceMetric::L2))
    );
    if (!supports_distance(metric_)) {
        throw std::runtime_error("FlatGPUANNS: unsupported distance metric");
    }

    uint32_t requested_dim = params.get<uint32_t>("vecDim", 0);
    if (!dataset.empty()) {
        dimension_ = static_cast<uint32_t>(dataset.front().second.size());
    } else {
        dimension_ = requested_dim;
    }

    if (dimension_ == 0) {
        throw std::runtime_error("FlatGPUANNS: dimension must be > 0");
    }

    for (const auto& entry : dataset) {
        if (entry.second.size() != dimension_) {
            throw std::runtime_error("FlatGPUANNS: inconsistent vector dimensions in dataset");
        }
    }

    auto host_cfg = extract_host_config(params, dataset.size());

    impl_->reset();
    impl_->init(dimension_, host_cfg.capacity);
    impl_->configure_host(host_cfg.mem_buffer_size,
                          host_cfg.dco_batch_size,
                          host_cfg.sketch_size,
                          host_cfg.amm_algo,
                          host_cfg.libamm_use_cuda);
    impl_->append_bulk(dataset);

    built_ = true;

    impl_->configure_cuda(dimension_, build_params_);
    impl_->mark_cuda_dirty();
    impl_->sync_cuda();

    auto end = std::chrono::high_resolution_clock::now();
    metrics_.build_time_seconds = std::chrono::duration<double>(end - start).count();
    metrics_.index_size_bytes = impl_->memory_usage_bytes();
}

bool FlatGPUANNS::save(const std::string& path) const {
    if (!built_) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    uint32_t magic = kFlatGPUSerializationMagic;
    uint32_t version = kFlatGPUSerializationVersion;
    uint32_t metric_val = static_cast<uint32_t>(metric_);
    uint32_t dim = dimension_;
    uint64_t count = impl_->size();

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&metric_val), sizeof(metric_val));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    const auto& ids = impl_->ids();
    const auto& raw = impl_->raw_data();
    out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(VectorId));
    out.write(reinterpret_cast<const char*>(raw.data()), raw.size() * sizeof(float));

    // Serialize build params for reproducibility
    uint64_t param_count = build_params_.params.size();
    out.write(reinterpret_cast<const char*>(&param_count), sizeof(param_count));
    for (const auto& [key, value] : build_params_.params) {
        uint64_t key_size = key.size();
        uint64_t value_size = value.size();
        out.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        out.write(key.data(), static_cast<std::streamsize>(key_size));
        out.write(reinterpret_cast<const char*>(&value_size), sizeof(value_size));
        out.write(value.data(), static_cast<std::streamsize>(value_size));
    }

    return true;
}

bool FlatGPUANNS::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t metric_val = 0;
    uint32_t dim = 0;
    uint64_t count = 0;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&metric_val), sizeof(metric_val));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    if (magic != kFlatGPUSerializationMagic || version != kFlatGPUSerializationVersion) {
        return false;
    }

    metric_ = static_cast<DistanceMetric>(metric_val);
    if (!supports_distance(metric_)) {
        return false;
    }
    dimension_ = dim;

    impl_->reset();
    impl_->init(dimension_, static_cast<size_t>(count));

    std::vector<VectorId> ids(count);
    in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(VectorId));

    std::vector<float> raw(static_cast<size_t>(count) * dimension_);
    in.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float));

    if (!in) {
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        Vector vec(dimension_);
        std::copy_n(raw.data() + i * dimension_, dimension_, vec.begin());
        impl_->append(ids[i], vec);
    }

    uint64_t param_count = 0;
    if (in.read(reinterpret_cast<char*>(&param_count), sizeof(param_count))) {
        build_params_.params.clear();
        for (uint64_t i = 0; i < param_count; ++i) {
            uint64_t key_size = 0;
            uint64_t value_size = 0;
            in.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
            std::string key(key_size, '\0');
            in.read(key.data(), static_cast<std::streamsize>(key_size));
            in.read(reinterpret_cast<char*>(&value_size), sizeof(value_size));
            std::string value(value_size, '\0');
            in.read(value.data(), static_cast<std::streamsize>(value_size));
            build_params_.set_raw(key, value);
        }
    }

    auto host_cfg = extract_host_config(build_params_, static_cast<size_t>(count));
    impl_->configure_host(host_cfg.mem_buffer_size,
                          host_cfg.dco_batch_size,
                          host_cfg.sketch_size,
                          host_cfg.amm_algo,
                          host_cfg.libamm_use_cuda);

    built_ = true;
    metrics_.reset();
    metrics_.index_size_bytes = impl_->memory_usage_bytes();

    impl_->configure_cuda(dimension_, build_params_);
    impl_->mark_cuda_dirty();
    impl_->sync_cuda();
    return true;
}

static bool is_better(DistanceMetric metric, float lhs, float rhs) {
    if (metric == DistanceMetric::INNER_PRODUCT) {
        return lhs > rhs;
    }
    return lhs < rhs;
}

#ifdef ENABLE_LIBAMM
struct LibAMMEngine {
    LibAMMEngine()
        : table_(std::make_unique<LibAMM::CPPAlgoTable>()) {}

    LibAMM::AbstractCPPAlgo* algo(const std::string& tag) {
        auto ptr = table_->findCppAlgo(tag);
        return ptr ? ptr.get() : nullptr;
    }

    at::Tensor run(const std::string& tag,
                   const at::Tensor& A,
                   const at::Tensor& B,
                   uint64_t sketch_size,
                   bool use_cuda) {
        auto ptr = algo(tag);
        if (!ptr) {
            throw std::runtime_error("LibAMMEngine: requested algorithm not found: " + tag);
        }
        auto cfg = newConfigMap();
        cfg->edit("useCuda", static_cast<uint64_t>(use_cuda));
        cfg->edit("sketchSize", static_cast<uint64_t>(sketch_size));
        ptr->setConfig(cfg);
        return ptr->amm(A, B, sketch_size);
    }

private:
    std::unique_ptr<LibAMM::CPPAlgoTable> table_;
};

static LibAMMEngine& libamm_engine() {
    static LibAMMEngine engine;
    return engine;
}
#endif

ANNSResult FlatGPUANNS::query(const Vector& query_vector,
                              const QueryConfig& config) const {
    if (!built_) {
        throw std::runtime_error("FlatGPUANNS: index not built");
    }
    if (query_vector.size() != dimension_) {
        throw std::runtime_error("FlatGPUANNS: query dimension mismatch");
    }

    auto start = std::chrono::high_resolution_clock::now();

    const size_t n = impl_->size();
    const size_t k = std::min(static_cast<size_t>(config.k), n);

    if (n == 0 || k == 0) {
        return ANNSResult();
    }

    bool prefer_libamm = false;
#ifdef ENABLE_LIBAMM
    prefer_libamm = (impl_->amm_algo() == "crs" || impl_->amm_algo() == "smp-pca") &&
                    impl_->sketch_size() > 0;
#endif

    bool want_gpu = impl_->using_cuda() &&
        config.algorithm_params.get<bool>("useGPU", true) &&
        !prefer_libamm;

#ifdef ENABLE_LIBAMM
    at::Tensor libamm_query_tensor;
    at::Tensor libamm_query_t;
    float libamm_query_norm_sq = 0.0f;
    float libamm_query_norm = 0.0f;
    bool libamm_use_cuda = false;
    const bool libamm_needs_norms = (metric_ == DistanceMetric::L2 || metric_ == DistanceMetric::COSINE);
    if (prefer_libamm) {
        libamm_use_cuda = impl_->libamm_use_cuda();
        if (config.algorithm_params.has("libammUseCuda")) {
            libamm_use_cuda = config.algorithm_params.get<bool>("libammUseCuda", libamm_use_cuda);
        }
        libamm_query_tensor = tensor_from_span(query_vector.data(), 1, dimension_, at::kCPU);
        libamm_query_t = libamm_query_tensor.transpose(0, 1).contiguous();
        if (libamm_needs_norms) {
            libamm_query_norm_sq = libamm_query_tensor.mul(libamm_query_tensor).sum().item<float>();
            if (libamm_query_norm_sq < 0.0f) {
                libamm_query_norm_sq = 0.0f;
            }
            libamm_query_norm = std::sqrt(libamm_query_norm_sq);
        }
    }
#endif

    if (want_gpu) {
        try {
            impl_->sync_cuda();

            auto& buffers = impl_->device_buffers();
            auto& scratch = impl_->query_scratch();

            flat_gpu::DeviceStats ensure_stats;
            impl_->cuda_backend()->ensure_query_capacity(buffers, scratch, 1, k, dimension_, ensure_stats);

            std::vector<VectorId> topk_ids(k, 0);
            std::vector<float> topk_distances(config.return_distances ? k : 0);

            float* distances_ptr = config.return_distances ? topk_distances.data() : nullptr;
            flat_gpu::DeviceStats query_stats;
            impl_->cuda_backend()->run_query(buffers,
                                             scratch,
                                             impl_->ids(),
                                             query_vector.data(),
                                             1,
                                             k,
                                             metric_,
                                             config.return_distances,
                                             distances_ptr,
                                             topk_ids.data(),
                                             query_stats);
            impl_->set_last_query_stats(query_stats);

            ANNSResult result;
            result.ids = std::move(topk_ids);
            if (config.return_distances) {
                result.distances = std::move(topk_distances);
            }
            result.actual_k = k;

            auto end = std::chrono::high_resolution_clock::now();
            metrics_.search_time_seconds += std::chrono::duration<double>(end - start).count();
            metrics_.distance_computations += n;

            return result;
        } catch (const std::exception&) {
            // Fallback to CPU path if GPU invocation fails
        }
    }

    size_t chunk = impl_->dco_batch_size();
    if (chunk == 0) {
        chunk = n;
    }
    chunk = std::max<size_t>(chunk, 1);

    auto worst_cmp = [this](const std::pair<float, VectorId>& a,
                             const std::pair<float, VectorId>& b) {
        return is_better(metric_, a.first, b.first);
    };

    std::priority_queue<
        std::pair<float, VectorId>,
        std::vector<std::pair<float, VectorId>>,
        decltype(worst_cmp)>
        topk_heap(worst_cmp);

    size_t processed = 0;
    while (processed < n) {
        size_t end = std::min(processed + chunk, n);

#ifdef ENABLE_LIBAMM
        std::vector<float> libamm_distances;
        bool libamm_chunk_ok = false;
        if (prefer_libamm) {
            try {
                const size_t rows = end - processed;
                if (rows > 0) {
                    const float* base_ptr = impl_->raw_data().data() + processed * dimension_;
                    auto db_tensor = tensor_from_span(base_ptr, rows, dimension_, at::kCPU);
                    const uint64_t sketch_size = std::max<uint64_t>(1, impl_->sketch_size());
                    auto result = libamm_engine().run(impl_->amm_algo(), db_tensor, libamm_query_t, sketch_size, libamm_use_cuda);
                    auto dot_tensor = result.to(at::kCPU).contiguous().view({static_cast<long>(rows)});
                    auto dot_ptr = dot_tensor.data_ptr<float>();

                    at::Tensor x_norms_tensor;
                    const float* x_norms_ptr = nullptr;
                    if (libamm_needs_norms) {
                        x_norms_tensor = db_tensor.mul(db_tensor).sum(1).contiguous();
                        x_norms_ptr = x_norms_tensor.data_ptr<float>();
                    }

                    libamm_distances.resize(rows);
                    switch (metric_) {
                        case DistanceMetric::L2: {
                            if (!x_norms_ptr) {
                                throw std::runtime_error("LibAMM: missing norms for L2 metric");
                            }
                            for (size_t j = 0; j < rows; ++j) {
                                float sq = x_norms_ptr[j] + libamm_query_norm_sq - 2.0f * dot_ptr[j];
                                if (sq < 0.0f) {
                                    sq = 0.0f;
                                }
                                libamm_distances[j] = std::sqrt(sq);
                            }
                            break;
                        }
                        case DistanceMetric::INNER_PRODUCT: {
                            for (size_t j = 0; j < rows; ++j) {
                                libamm_distances[j] = dot_ptr[j];
                            }
                            break;
                        }
                        case DistanceMetric::COSINE: {
                            if (!x_norms_ptr) {
                                throw std::runtime_error("LibAMM: missing norms for cosine metric");
                            }
                            for (size_t j = 0; j < rows; ++j) {
                                float x_norm = std::sqrt(std::max(0.0f, x_norms_ptr[j]));
                                float denom = libamm_query_norm * x_norm;
                                if (denom == 0.0f) {
                                    libamm_distances[j] = 1.0f;
                                } else {
                                    float cos_sim = dot_ptr[j] / denom;
                                    cos_sim = std::clamp(cos_sim, -1.0f, 1.0f);
                                    libamm_distances[j] = 1.0f - cos_sim;
                                }
                            }
                            break;
                        }
                    }
                    libamm_chunk_ok = true;
                }
            } catch (const std::exception&) {
                // fall back to exact computation for this chunk
            }
        }
#endif

        for (size_t i = processed; i < end; ++i) {
            const float* row = impl_->row_ptr(i);
            if (row == nullptr) {
                continue;
            }
            float dist;
#ifdef ENABLE_LIBAMM
            if (prefer_libamm && libamm_chunk_ok) {
                dist = libamm_distances[i - processed];
            } else
#endif
            {
                dist = compute_distance(metric_, row, query_vector, dimension_);
            }

            if (topk_heap.size() < k) {
                topk_heap.emplace(dist, impl_->ids()[i]);
            } else if (is_better(metric_, dist, topk_heap.top().first)) {
                topk_heap.pop();
                topk_heap.emplace(dist, impl_->ids()[i]);
            }
        }
        processed = end;
    }

    std::vector<std::pair<float, VectorId>> scored;
    scored.reserve(topk_heap.size());
    while (!topk_heap.empty()) {
        scored.push_back(topk_heap.top());
        topk_heap.pop();
    }

    auto comparator = [this](const auto& a, const auto& b) {
        return is_better(metric_, a.first, b.first);
    };
    std::sort(scored.begin(), scored.end(), comparator);

    ANNSResult result;
    result.ids.reserve(scored.size());
    if (config.return_distances) {
        result.distances.reserve(scored.size());
    }
    for (const auto& entry : scored) {
        result.ids.push_back(entry.second);
        if (config.return_distances) {
            result.distances.push_back(entry.first);
        }
    }
    result.actual_k = scored.size();

    auto end = std::chrono::high_resolution_clock::now();
    metrics_.search_time_seconds += std::chrono::duration<double>(end - start).count();
    metrics_.distance_computations += n;

    return result;
}

std::vector<ANNSResult> FlatGPUANNS::batch_query(
    const std::vector<Vector>& query_vectors,
    const QueryConfig& config) const {
    std::vector<ANNSResult> results;
    results.reserve(query_vectors.size());
    for (const auto& query : query_vectors) {
        results.push_back(this->query(query, config));
    }
    return results;
}

void FlatGPUANNS::add_vector(const VectorEntry& entry) {
    if (!built_) {
        throw std::runtime_error("FlatGPUANNS: index not built");
    }
    if (entry.second.size() != dimension_) {
        throw std::runtime_error("FlatGPUANNS: vector dimension mismatch on add");
    }
    impl_->append(entry.first, entry.second);
    metrics_.index_size_bytes = impl_->memory_usage_bytes();
    impl_->mark_cuda_dirty();
}

void FlatGPUANNS::add_vectors(const std::vector<VectorEntry>& entries) {
    for (const auto& entry : entries) {
        add_vector(entry);
    }
}

void FlatGPUANNS::remove_vector(VectorId id) {
    if (!built_) {
        throw std::runtime_error("FlatGPUANNS: index not built");
    }
    impl_->remove(id);
    metrics_.index_size_bytes = impl_->memory_usage_bytes();
    impl_->mark_cuda_dirty();
}

void FlatGPUANNS::remove_vectors(const std::vector<VectorId>& ids) {
    for (auto id : ids) {
        remove_vector(id);
    }
}

size_t FlatGPUANNS::get_index_size() const {
    return impl_->size();
}

size_t FlatGPUANNS::get_memory_usage() const {
    return impl_->memory_usage_bytes();
}

std::unordered_map<std::string, std::string> FlatGPUANNS::get_build_params() const {
    return build_params_.params;
}

ANNSMetrics FlatGPUANNS::get_metrics() const {
    ANNSMetrics metrics = metrics_;
    metrics.additional_metrics["host_memory_bytes"] = static_cast<double>(impl_->memory_usage_bytes());
    metrics.additional_metrics["memory_read_cnt_total"] = static_cast<double>(impl_->memory_read_cnt_total());
    metrics.additional_metrics["memory_read_cnt_miss"] = static_cast<double>(impl_->memory_read_cnt_miss());
    metrics.additional_metrics["memory_write_cnt_total"] = static_cast<double>(impl_->memory_write_cnt_total());
    metrics.additional_metrics["memory_write_cnt_miss"] = static_cast<double>(impl_->memory_write_cnt_miss());
    metrics.additional_metrics["gpu_enabled"] = impl_->using_cuda() ? 1.0 : 0.0;
    metrics.additional_metrics["gpu_last_upload_ms"] = impl_->last_upload_stats().upload_ms;
    metrics.additional_metrics["gpu_last_compute_ms"] = impl_->last_query_stats().compute_ms;
    metrics.additional_metrics["gpu_last_download_ms"] = impl_->last_query_stats().download_ms;
    metrics.additional_metrics["mem_buffer_size"] = static_cast<double>(impl_->mem_buffer_size());
    metrics.additional_metrics["dco_batch_size"] = static_cast<double>(impl_->dco_batch_size());
    metrics.additional_metrics["sketch_size"] = static_cast<double>(impl_->sketch_size());
    return metrics;
}

bool FlatGPUANNS::validate_params(const AlgorithmParams& params) const {
    if (!supports_distance(static_cast<DistanceMetric>(
            params.get<int>("metric", static_cast<int>(DistanceMetric::L2))))) {
        return false;
    }
    auto batch_size = params.get<size_t>("DCOBatchSize", 0);
    if (batch_size == 0 && params.has("DCOBatchSize")) {
        return false;
    }
    auto mem_buffer = params.get<size_t>("memBufferSize", 0);
    if (mem_buffer == 0 && params.has("memBufferSize")) {
        return false;
    }
    auto sketch_size = params.get<size_t>("sketchSize", 0);
    if (sketch_size == 0 && params.has("sketchSize")) {
        return false;
    }
    auto amm_algo = params.get<std::string>("ammAlgo", "mm");
    std::string normalized = amm_algo;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized != "mm" && normalized != "crs" && normalized != "smp-pca") {
        return false;
    }
    return true;
}

AlgorithmParams FlatGPUANNS::get_default_params() const {
    AlgorithmParams params;
    params.set("metric", static_cast<int>(DistanceMetric::L2));
    params.set("capacity", static_cast<size_t>(16384));
    params.set("memBufferSize", static_cast<size_t>(16384));
    params.set("sketchSize", static_cast<size_t>(10));
    params.set("DCOBatchSize", static_cast<size_t>(16384));
    params.set("cudaDevice", static_cast<int64_t>(-1));
    params.set("enableGPU", true);
    params.set("vecDim", static_cast<uint32_t>(0));
    params.set("ammAlgo", std::string("mm"));
    params.set("libammUseCuda", false);
    return params;
}

QueryConfig FlatGPUANNS::get_default_query_config() const {
    QueryConfig config;
    config.k = 10;
    config.return_distances = true;
    return config;
}

std::unique_ptr<ANNSAlgorithm> FlatGPUANNSFactory::create() const {
    return std::make_unique<FlatGPUANNS>();
}

std::string FlatGPUANNSFactory::algorithm_description() const {
    return "Flat brute-force ANN with host buffer and optional GPU execution";
}

std::vector<DistanceMetric> FlatGPUANNSFactory::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

AlgorithmParams FlatGPUANNSFactory::default_build_params() const {
    return FlatGPUANNS().get_default_params();
}

QueryConfig FlatGPUANNSFactory::default_query_config() const {
    return FlatGPUANNS().get_default_query_config();
}

} // namespace anns
} // namespace sage_vdb
