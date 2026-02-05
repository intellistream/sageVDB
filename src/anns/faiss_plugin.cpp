#include "sage_vdb/anns/faiss_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef ENABLE_FAISS
#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/impl/FaissAssert.h>
#endif

namespace sage_vdb {
namespace anns {

namespace {
#ifdef ENABLE_FAISS
REGISTER_ANNS_ALGORITHM(FaissANNSFactory);
#endif
}

FaissANNS::FaissANNS()
        : distance_metric_(DistanceMetric::L2),
            dimension_(0),
            index_type_(IndexType::AUTO),
            last_index_size_bytes_(0),
            is_built_(false) {
    metrics_.reset();
}

FaissANNS::~FaissANNS() = default;

std::string FaissANNS::version() const {
#ifdef ENABLE_FAISS
    std::ostringstream oss;
    oss << "FAISS " << FAISS_VERSION_MAJOR << '.' << FAISS_VERSION_MINOR;
#ifdef FAISS_VERSION_PATCH
    oss << '.' << FAISS_VERSION_PATCH;
#endif
    return oss.str();
#else
    return "unavailable";
#endif
}

std::string FaissANNS::description() const {
    return "FAISS-based approximate nearest neighbor search";
}

std::vector<DistanceMetric> FaissANNS::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

bool FaissANNS::supports_distance(DistanceMetric metric) const {
    auto supported = supported_distances();
    return std::find(supported.begin(), supported.end(), metric) != supported.end();
}

void FaissANNS::fit(const std::vector<VectorEntry>& dataset,
                    const AlgorithmParams& params) {
#ifndef ENABLE_FAISS
    (void)dataset;
    (void)params;
    throw std::runtime_error("FAISS support not enabled in this build");
#else
    metrics_.reset();

    if (dataset.empty()) {
        index_.reset();
        dimension_ = 0;
        is_built_ = true;
        build_params_ = params;
        last_index_size_bytes_ = 0;
        metrics_.index_size_bytes = 0;
        return;
    }

    dimension_ = static_cast<int>(dataset.front().second.size());
    for (const auto& entry : dataset) {
        if (static_cast<int>(entry.second.size()) != dimension_) {
            throw std::runtime_error("FaissANNS: inconsistent vector dimensions");
        }
    }

    build_params_ = params;
    distance_metric_ = parse_distance_metric(params);
    index_type_ = parse_index_type(params.get<std::string>("index_type", "auto"));
    build_params_.set("metric", static_cast<int>(distance_metric_));
    build_params_.set_raw("metric_name", metric_to_string(distance_metric_));
    build_params_.set_raw("index_type", index_type_to_string(index_type_));

    auto make_index = [&]() -> std::unique_ptr<faiss::Index> {
        switch (index_type_) {
            case IndexType::FLAT:
                return create_flat_index(dimension_);
            case IndexType::IVF_FLAT:
                return create_ivf_flat_index(dimension_, params);
            case IndexType::IVF_PQ:
                return create_ivf_pq_index(dimension_, params);
            case IndexType::HNSW:
                return create_hnsw_index(dimension_, params);
            case IndexType::AUTO:
            default:
                return create_auto_index(dimension_, dataset.size(), params);
        }
    };

    index_ = make_index();
    if (!index_) {
        throw std::runtime_error("FaissANNS: failed to create FAISS index");
    }

    if (index_type_ == IndexType::AUTO) {
        if (dynamic_cast<faiss::IndexIVFPQ*>(index_.get())) {
            index_type_ = IndexType::IVF_PQ;
        } else if (dynamic_cast<faiss::IndexIVFFlat*>(index_.get())) {
            index_type_ = IndexType::IVF_FLAT;
        } else if (dynamic_cast<faiss::IndexHNSW*>(index_.get())) {
            index_type_ = IndexType::HNSW;
        } else {
            index_type_ = IndexType::FLAT;
        }
    }
    build_params_.set_raw("resolved_index_type", index_type_to_string(index_type_));
    last_index_size_bytes_ = 0;

    std::vector<float> data(dataset.size() * static_cast<size_t>(dimension_));
    std::vector<faiss::idx_t> ids(dataset.size());

    bool normalize = distance_metric_ == DistanceMetric::COSINE;
    for (size_t i = 0; i < dataset.size(); ++i) {
        ids[i] = static_cast<faiss::idx_t>(dataset[i].first);
        std::vector<float> tmp = dataset[i].second;
        if (static_cast<int>(tmp.size()) != dimension_) {
            throw std::runtime_error("FaissANNS: vector dimension mismatch during build");
        }
        if (normalize) {
            normalize_vector(tmp);
        }
        std::copy(tmp.begin(), tmp.end(), data.begin() + i * static_cast<size_t>(dimension_));
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (!index_->is_trained) {
        index_->train(dataset.size(), data.data());
    }

    index_->add_with_ids(dataset.size(), data.data(), ids.data());

    auto end = std::chrono::high_resolution_clock::now();
    metrics_.build_time_seconds = std::chrono::duration<double>(end - start).count();
    last_index_size_bytes_ = dataset.size() * static_cast<size_t>(dimension_) * sizeof(float);
    metrics_.index_size_bytes = last_index_size_bytes_;
    is_built_ = true;
#endif
}

bool FaissANNS::save(const std::string& path) const {
#ifndef ENABLE_FAISS
    (void)path;
    return false;
#else
    if (!index_) {
        return false;
    }
    try {
        faiss::write_index(index_.get(), path.c_str());

        std::ofstream meta(metadata_path(path), std::ios::binary);
        if (meta.is_open()) {
            int metric = static_cast<int>(distance_metric_);
            int type = static_cast<int>(index_type_);
            meta.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
            meta.write(reinterpret_cast<const char*>(&type), sizeof(type));
            uint32_t param_count = static_cast<uint32_t>(build_params_.params.size());
            meta.write(reinterpret_cast<const char*>(&param_count), sizeof(param_count));
            for (const auto& [key, value] : build_params_.params) {
                uint32_t klen = static_cast<uint32_t>(key.size());
                uint32_t vlen = static_cast<uint32_t>(value.size());
                meta.write(reinterpret_cast<const char*>(&klen), sizeof(klen));
                meta.write(key.data(), klen);
                meta.write(reinterpret_cast<const char*>(&vlen), sizeof(vlen));
                meta.write(value.data(), vlen);
            }
            meta.close();
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
#endif
}

bool FaissANNS::load(const std::string& path) {
#ifndef ENABLE_FAISS
    (void)path;
    return false;
#else
    try {
        index_.reset(faiss::read_index(path.c_str()));
        if (!index_) {
            return false;
        }
        dimension_ = static_cast<int>(index_->d);
        is_built_ = true;
        build_params_ = get_default_params();

        // Default metric inference from index
        switch (index_->metric_type) {
            case faiss::MetricType::METRIC_L2:
                distance_metric_ = DistanceMetric::L2;
                break;
            case faiss::MetricType::METRIC_INNER_PRODUCT:
                distance_metric_ = DistanceMetric::INNER_PRODUCT;
                break;
            default:
                distance_metric_ = DistanceMetric::L2;
                break;
        }

        if (dynamic_cast<faiss::IndexIVFPQ*>(index_.get())) {
            index_type_ = IndexType::IVF_PQ;
        } else if (dynamic_cast<faiss::IndexIVFFlat*>(index_.get())) {
            index_type_ = IndexType::IVF_FLAT;
        } else if (dynamic_cast<faiss::IndexHNSW*>(index_.get())) {
            index_type_ = IndexType::HNSW;
        } else {
            index_type_ = IndexType::FLAT;
        }
        build_params_.set_raw("resolved_index_type", index_type_to_string(index_type_));

        // Attempt to read metadata
        std::ifstream meta(metadata_path(path), std::ios::binary);
        if (meta.is_open()) {
            int metric = 0;
            int type = 0;
            meta.read(reinterpret_cast<char*>(&metric), sizeof(metric));
            meta.read(reinterpret_cast<char*>(&type), sizeof(type));
            distance_metric_ = static_cast<DistanceMetric>(metric);
            index_type_ = static_cast<IndexType>(type);
            build_params_.params.clear();
            uint32_t param_count = 0;
            meta.read(reinterpret_cast<char*>(&param_count), sizeof(param_count));
            for (uint32_t i = 0; i < param_count; ++i) {
                uint32_t klen = 0;
                uint32_t vlen = 0;
                meta.read(reinterpret_cast<char*>(&klen), sizeof(klen));
                std::string key(klen, '\0');
                meta.read(key.data(), klen);
                meta.read(reinterpret_cast<char*>(&vlen), sizeof(vlen));
                std::string value(vlen, '\0');
                meta.read(value.data(), vlen);
                build_params_.set_raw(key, value);
            }
            meta.close();
        }

    build_params_.set("metric", static_cast<int>(distance_metric_));
    build_params_.set_raw("metric_name", metric_to_string(distance_metric_));
    build_params_.set_raw("index_type", index_type_to_string(index_type_));
    build_params_.set_raw("resolved_index_type", index_type_to_string(index_type_));

    metrics_.reset();
    last_index_size_bytes_ = static_cast<size_t>(index_->ntotal) * static_cast<size_t>(dimension_) * sizeof(float);
    metrics_.index_size_bytes = last_index_size_bytes_;
        return true;
    } catch (const std::exception&) {
        index_.reset();
        is_built_ = false;
        return false;
    }
#endif
}

bool FaissANNS::is_built() const {
#ifdef ENABLE_FAISS
    return is_built_ && index_ != nullptr;
#else
    return false;
#endif
}

ANNSResult FaissANNS::query(const Vector& query_vector,
                            const QueryConfig& config) const {
#ifndef ENABLE_FAISS
    (void)query_vector;
    (void)config;
    throw std::runtime_error("FAISS support not enabled in this build");
#else
    if (!index_ || !is_built_) {
        throw std::runtime_error("FaissANNS index is not built");
    }

    uint32_t k = std::min(config.k, static_cast<uint32_t>(index_->ntotal));
    if (k == 0) {
        return {};
    }

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);

    std::vector<float> query = query_vector;
    if (distance_metric_ == DistanceMetric::COSINE) {
        normalize_vector(query);
    }

    auto start = std::chrono::high_resolution_clock::now();
    apply_query_params(config);
    index_->search(1, query.data(), k, distances.data(), labels.data());
    auto end = std::chrono::high_resolution_clock::now();

    metrics_.search_time_seconds = std::chrono::duration<double>(end - start).count();
    metrics_.distance_computations = k;

    auto result = convert_faiss_results(labels.data(), distances.data(), k);
    if (!config.return_distances) {
        result.distances.clear();
    }
    return result;
#endif
}

std::vector<ANNSResult> FaissANNS::batch_query(
    const std::vector<Vector>& query_vectors,
    const QueryConfig& config) const {
#ifndef ENABLE_FAISS
    (void)query_vectors;
    (void)config;
    throw std::runtime_error("FAISS support not enabled in this build");
#else
    if (!index_ || !is_built_) {
        throw std::runtime_error("FaissANNS index is not built");
    }

    if (query_vectors.empty()) {
        return {};
    }

    uint32_t k = std::min(config.k, static_cast<uint32_t>(index_->ntotal));
    if (k == 0) {
        return std::vector<ANNSResult>(query_vectors.size());
    }

    size_t nq = query_vectors.size();
    std::vector<float> queries(nq * static_cast<size_t>(dimension_));

    for (size_t i = 0; i < nq; ++i) {
        if (query_vectors[i].size() != static_cast<size_t>(dimension_)) {
            throw std::runtime_error("FaissANNS: query dimension mismatch");
        }
        std::vector<float> tmp = query_vectors[i];
        if (distance_metric_ == DistanceMetric::COSINE) {
            normalize_vector(tmp);
        }
        std::copy(tmp.begin(), tmp.end(), queries.begin() + i * static_cast<size_t>(dimension_));
    }

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    auto start = std::chrono::high_resolution_clock::now();
    apply_query_params(config);
    index_->search(nq, queries.data(), k, distances.data(), labels.data());
    auto end = std::chrono::high_resolution_clock::now();

    metrics_.search_time_seconds = std::chrono::duration<double>(end - start).count();
    metrics_.distance_computations = nq * k;

    std::vector<ANNSResult> results;
    results.reserve(nq);
    for (size_t i = 0; i < nq; ++i) {
        const auto* label_ptr = labels.data() + i * k;
        const auto* dist_ptr = distances.data() + i * k;
        auto res = convert_faiss_results(label_ptr, dist_ptr, k);
        if (!config.return_distances) {
            res.distances.clear();
        }
        results.emplace_back(std::move(res));
    }
    return results;
#endif
}

ANNSResult FaissANNS::range_query(const Vector& query_vector,
                                  float radius,
                                  const QueryConfig& config) const {
#ifndef ENABLE_FAISS
    (void)query_vector;
    (void)radius;
    (void)config;
    throw std::runtime_error("FAISS support not enabled in this build");
#else
    if (!index_ || !is_built_) {
        throw std::runtime_error("FaissANNS index is not built");
    }

    std::vector<float> query = query_vector;
    if (distance_metric_ == DistanceMetric::COSINE) {
        normalize_vector(query);
    }

    faiss::RangeSearchResult result_container(1);
    auto start = std::chrono::high_resolution_clock::now();
    apply_query_params(config);
    index_->range_search(1, query.data(), radius, &result_container);
    auto end = std::chrono::high_resolution_clock::now();

    metrics_.search_time_seconds = std::chrono::duration<double>(end - start).count();
    metrics_.distance_computations = result_container.lims[1] - result_container.lims[0];

    size_t from = result_container.lims[0];
    size_t to = result_container.lims[1];

    ANNSResult result;
    result.actual_k = to - from;
    result.ids.reserve(result.actual_k);
    if (config.return_distances) {
        result.distances.reserve(result.actual_k);
    }

    for (size_t i = from; i < to; ++i) {
        auto id = static_cast<VectorId>(result_container.labels[i]);
        result.ids.push_back(id);
        if (config.return_distances) {
            float dist = result_container.distances[i];
            if (distance_metric_ == DistanceMetric::COSINE) {
                dist = 1.0f - dist;
            }
            result.distances.push_back(dist);
        }
    }
    return result;
#endif
}

void FaissANNS::add_vector(const VectorEntry& entry) {
#ifndef ENABLE_FAISS
    (void)entry;
    throw std::runtime_error("FAISS support not enabled in this build");
#else
    if (!index_ || !is_built_) {
        throw std::runtime_error("FaissANNS index is not built");
    }
    if (static_cast<int>(entry.second.size()) != dimension_) {
        throw std::runtime_error("FaissANNS: vector dimension mismatch on add");
    }
    std::vector<float> tmp = entry.second;
    if (distance_metric_ == DistanceMetric::COSINE) {
        normalize_vector(tmp);
    }
    faiss::idx_t id = static_cast<faiss::idx_t>(entry.first);
    index_->add_with_ids(1, tmp.data(), &id);
    last_index_size_bytes_ = static_cast<size_t>(index_->ntotal) * static_cast<size_t>(dimension_) * sizeof(float);
    metrics_.index_size_bytes = last_index_size_bytes_;
#endif
}

void FaissANNS::add_vectors(const std::vector<VectorEntry>& entries) {
#ifndef ENABLE_FAISS
    (void)entries;
    throw std::runtime_error("FAISS support not enabled in this build");
#else
    if (entries.empty()) {
        return;
    }
    std::vector<float> data(entries.size() * static_cast<size_t>(dimension_));
    std::vector<faiss::idx_t> ids(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        if (static_cast<int>(entries[i].second.size()) != dimension_) {
            throw std::runtime_error("FaissANNS: vector dimension mismatch on batch add");
        }
        std::vector<float> tmp = entries[i].second;
        if (distance_metric_ == DistanceMetric::COSINE) {
            normalize_vector(tmp);
        }
        std::copy(tmp.begin(), tmp.end(), data.begin() + i * static_cast<size_t>(dimension_));
        ids[i] = static_cast<faiss::idx_t>(entries[i].first);
    }
    index_->add_with_ids(entries.size(), data.data(), ids.data());
    last_index_size_bytes_ = static_cast<size_t>(index_->ntotal) * static_cast<size_t>(dimension_) * sizeof(float);
    metrics_.index_size_bytes = last_index_size_bytes_;
#endif
}

size_t FaissANNS::get_index_size() const {
#ifdef ENABLE_FAISS
    return index_ ? static_cast<size_t>(index_->ntotal) : 0;
#else
    return 0;
#endif
}

size_t FaissANNS::get_memory_usage() const {
#ifdef ENABLE_FAISS
    return last_index_size_bytes_;
#else
    return 0;
#endif
}

std::unordered_map<std::string, std::string> FaissANNS::get_build_params() const {
    return build_params_.params;
}

ANNSMetrics FaissANNS::get_metrics() const {
    return metrics_;
}

bool FaissANNS::validate_params(const AlgorithmParams& params) const {
    auto type = parse_index_type(params.get<std::string>("index_type", "auto"));

    auto require_positive = [&](const std::string& key, int default_value) {
        int value = params.get<int>(key, default_value);
        return value > 0;
    };

    switch (type) {
        case IndexType::IVF_FLAT:
        case IndexType::IVF_PQ:
            if (!require_positive("nlist", 1024)) return false;
            if (type == IndexType::IVF_PQ) {
                if (!require_positive("m", 16)) return false;
                if (!require_positive("nbits", 8)) return false;
            }
            break;
        default:
            break;
    }
    return true;
}

AlgorithmParams FaissANNS::get_default_params() const {
    AlgorithmParams params;
    params.set("metric", static_cast<int>(DistanceMetric::L2));
    params.set("index_type", std::string("auto"));
    params.set("nlist", 1024);
    params.set("m", 16);
    params.set("nbits", 8);
    params.set("M", 32);
    params.set("efConstruction", 200);
    return params;
}

QueryConfig FaissANNS::get_default_query_config() const {
    QueryConfig config;
    config.k = 10;
    config.return_distances = true;
    config.algorithm_params.set("nprobe", 8);
    config.algorithm_params.set("efSearch", 64);
    return config;
}

std::unique_ptr<ANNSAlgorithm> FaissANNSFactory::create() const {
#ifdef ENABLE_FAISS
    return std::make_unique<FaissANNS>();
#else
    throw std::runtime_error("FAISS support not enabled in this build");
#endif
}

std::string FaissANNSFactory::algorithm_description() const {
    return "Facebook AI Similarity Search (FAISS)";
}

std::vector<DistanceMetric> FaissANNSFactory::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

AlgorithmParams FaissANNSFactory::default_build_params() const {
    return FaissANNS().get_default_params();
}

QueryConfig FaissANNSFactory::default_query_config() const {
    return FaissANNS().get_default_query_config();
}

#ifdef ENABLE_FAISS
std::unique_ptr<faiss::Index> FaissANNS::create_flat_index(int dim) {
    if (distance_metric_ == DistanceMetric::L2 || distance_metric_ == DistanceMetric::COSINE) {
        return std::unique_ptr<faiss::Index>(new faiss::IndexFlatL2(dim));
    }
    return std::unique_ptr<faiss::Index>(new faiss::IndexFlatIP(dim));
}

std::unique_ptr<faiss::Index> FaissANNS::create_ivf_flat_index(int dim, const AlgorithmParams& params) {
    int nlist = params.get<int>("nlist", 1024);
    if (nlist <= 0) {
        nlist = 1024;
    }
    build_params_.set("nlist", nlist);
    faiss::MetricType metric = (distance_metric_ == DistanceMetric::L2 || distance_metric_ == DistanceMetric::COSINE)
                                   ? faiss::MetricType::METRIC_L2
                                   : faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::Index* quantizer = nullptr;
    if (metric == faiss::MetricType::METRIC_L2) {
        quantizer = new faiss::IndexFlatL2(dim);
    } else {
        quantizer = new faiss::IndexFlatIP(dim);
    }
    auto index = std::unique_ptr<faiss::Index>(new faiss::IndexIVFFlat(quantizer, dim, nlist, metric));
    index->metric_type = metric;
    return index;
}

std::unique_ptr<faiss::Index> FaissANNS::create_ivf_pq_index(int dim, const AlgorithmParams& params) {
    int nlist = params.get<int>("nlist", 1024);
    int m = params.get<int>("m", 16);
    int nbits = params.get<int>("nbits", 8);
    if (nlist <= 0) nlist = 1024;
    if (m <= 0) m = 16;
    if (nbits <= 0) nbits = 8;
    build_params_.set("nlist", nlist);
    build_params_.set("m", m);
    build_params_.set("nbits", nbits);
    faiss::MetricType metric = (distance_metric_ == DistanceMetric::L2 || distance_metric_ == DistanceMetric::COSINE)
                                   ? faiss::MetricType::METRIC_L2
                                   : faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::Index* quantizer = nullptr;
    if (metric == faiss::MetricType::METRIC_L2) {
        quantizer = new faiss::IndexFlatL2(dim);
    } else {
        quantizer = new faiss::IndexFlatIP(dim);
    }
    auto index = std::unique_ptr<faiss::Index>(
        new faiss::IndexIVFPQ(quantizer, dim, nlist, m, nbits, metric));
    index->metric_type = metric;
    return index;
}

std::unique_ptr<faiss::Index> FaissANNS::create_hnsw_index(int dim, const AlgorithmParams& params) {
    int M = params.get<int>("M", 32);
    int efConstruction = params.get<int>("efConstruction", 200);
    build_params_.set("M", M);
    build_params_.set("efConstruction", efConstruction);
    auto metric = (distance_metric_ == DistanceMetric::L2 || distance_metric_ == DistanceMetric::COSINE)
                      ? faiss::MetricType::METRIC_L2
                      : faiss::MetricType::METRIC_INNER_PRODUCT;
    auto index = std::unique_ptr<faiss::Index>(new faiss::IndexHNSWFlat(dim, M, metric));
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());
    if (hnsw) {
        hnsw->hnsw.efConstruction = efConstruction;
    }
    return index;
}

std::unique_ptr<faiss::Index> FaissANNS::create_auto_index(int dim, size_t num_vectors,
                                                           const AlgorithmParams& params) {
    size_t threshold_ivf = params.get<int>("auto_threshold_ivf", 50000);
    build_params_.set("auto_threshold_ivf", static_cast<int>(threshold_ivf));
    if (num_vectors < threshold_ivf) {
        return create_flat_index(dim);
    }
    return create_ivf_flat_index(dim, params);
}
#endif

FaissANNS::IndexType FaissANNS::parse_index_type(const std::string& type_str) const {
    std::string lowered = type_str;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
    if (lowered == "flat") return IndexType::FLAT;
    if (lowered == "ivf_flat" || lowered == "ivfflat" || lowered == "ivf-flat") return IndexType::IVF_FLAT;
    if (lowered == "ivf_pq" || lowered == "ivfpq" || lowered == "ivf-pq") return IndexType::IVF_PQ;
    if (lowered == "hnsw") return IndexType::HNSW;
    return IndexType::AUTO;
}

DistanceMetric FaissANNS::parse_distance_metric(const AlgorithmParams& params) const {
    if (params.has("metric")) {
        const auto& raw = params.params.at("metric");
        std::string lowered = raw;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
        if (lowered == "l2" || lowered == "euclidean") return DistanceMetric::L2;
        if (lowered == "ip" || lowered == "inner" || lowered == "inner_product") return DistanceMetric::INNER_PRODUCT;
        if (lowered == "cosine") return DistanceMetric::COSINE;
        try {
            int metric_int = std::stoi(raw);
            return static_cast<DistanceMetric>(metric_int);
        } catch (const std::exception&) {
            // fall through to default
        }
    }
    return distance_metric_;
}

std::string FaissANNS::metric_to_string(DistanceMetric metric) const {
    switch (metric) {
        case DistanceMetric::L2:
            return "l2";
        case DistanceMetric::INNER_PRODUCT:
            return "inner_product";
        case DistanceMetric::COSINE:
            return "cosine";
        default:
            return "unknown";
    }
}

std::string FaissANNS::index_type_to_string(IndexType type) const {
    switch (type) {
        case IndexType::FLAT:
            return "flat";
        case IndexType::IVF_FLAT:
            return "ivf_flat";
        case IndexType::IVF_PQ:
            return "ivf_pq";
        case IndexType::HNSW:
            return "hnsw";
        case IndexType::AUTO:
        default:
            return "auto";
    }
}

ANNSResult FaissANNS::convert_faiss_results(const faiss::idx_t* ids,
                                            const float* distances,
                                            size_t k) const {
    ANNSResult result;
    result.actual_k = 0;
    result.ids.reserve(k);
    result.distances.reserve(k);

    for (size_t i = 0; i < k; ++i) {
        if (ids[i] < 0) {
            continue;
        }
        result.ids.push_back(static_cast<VectorId>(ids[i]));
        float dist = distances[i];
        if (distance_metric_ == DistanceMetric::COSINE) {
            dist = 1.0f - dist;
        }
        result.distances.push_back(dist);
        result.actual_k++;
    }
    return result;
}

void FaissANNS::normalize_vector(std::vector<float>& vec) const {
    float norm = 0.0f;
    for (float v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& v : vec) {
            v /= norm;
        }
    }
}

std::string FaissANNS::metadata_path(const std::string& base_path) const {
    return base_path + ".meta";
}

void FaissANNS::apply_query_params(const QueryConfig& config) const {
#ifdef ENABLE_FAISS
    if (!index_) {
        return;
    }

    if (config.algorithm_params.has("nprobe")) {
        int nprobe = config.algorithm_params.get<int>("nprobe", 0);
        if (nprobe > 0) {
            if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index_.get())) {
                ivf->nprobe = nprobe;
            }
        }
    }

    if (config.algorithm_params.has("efSearch")) {
        int ef = config.algorithm_params.get<int>("efSearch", 0);
        if (ef > 0) {
            if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index_.get())) {
                hnsw->hnsw.efSearch = ef;
            }
        }
    }
#endif
}

} // namespace anns
} // namespace sage_vdb
