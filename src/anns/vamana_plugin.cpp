#include "sage_vdb/anns/vamana_plugin.h"

#include "sage_vdb/anns/vamana/vertex.h"
#include "sage_vdb/anns/vamana/distance.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace sage_vdb {
namespace anns {

namespace {
REGISTER_ANNS_ALGORITHM(VamanaANNSFactory);
constexpr float kDefaultAlpha = 1.2f;
constexpr uint32_t kDeleteBatchThresholdPercent = 5;
}  // namespace

class VamanaANNS::Impl {
public:
    using DistAndId = std::pair<float, vamana::idx_t>;
    using MaxHeap = std::priority_queue<DistAndId, std::vector<DistAndId>, std::less<>>;
    using MinHeap = std::priority_queue<DistAndId, std::vector<DistAndId>, std::greater<>>;

    Impl()
        : metric(DistanceMetric::L2),
          dimension(0),
          entry_point(std::numeric_limits<vamana::idx_t>::max()),
          next_internal_id(0),
          M(8),
          Mmax(16),
          ef_construction(50),
          ef_search(200),
          alpha(kDefaultAlpha) {}

    void reset() {
        nodes.clear();
        delete_list.clear();
        id_map.clear();
        reverse_id_map.clear();
        dimension = 0;
        entry_point = std::numeric_limits<vamana::idx_t>::max();
        next_internal_id = 0;
    }

    float compute_distance(const Vector& a, const Vector& b) const {
        switch (metric) {
            case DistanceMetric::L2:
                return vamana::Distance::l2(a, b);
            case DistanceMetric::INNER_PRODUCT:
                return vamana::Distance::inner_product(a, b);
            case DistanceMetric::COSINE:
                return vamana::Distance::cosine(a, b);
            default:
                throw std::runtime_error("Vamana: unsupported distance metric");
        }
    }

    vamana::idx_t insert_node(VectorId external_id, const Vector& vector) {
        const vamana::idx_t internal_id = next_internal_id++;
        vamana::Vertex vertex(internal_id, vector);
        nodes.emplace(internal_id, std::move(vertex));
        id_map.emplace(external_id, internal_id);
        reverse_id_map.emplace(internal_id, external_id);

        if (entry_point == std::numeric_limits<vamana::idx_t>::max()) {
            entry_point = internal_id;
            return internal_id;
        }

        float nearest_dist = compute_distance(nodes.at(entry_point).vector, vector);
        vamana::idx_t nearest = entry_point;
        greedy_update_nearest(nearest, nearest_dist, vector);
        add_links_starting_from(internal_id, nearest);
        return internal_id;
    }

    void add_links_starting_from(vamana::idx_t start_id, vamana::idx_t nearest_id) {
        MaxHeap link_targets;
        greedy_search(nearest_id, nodes.at(start_id).vector, link_targets, ef_construction);
        shrink_neighbor_list(link_targets, Mmax);

        std::vector<vamana::idx_t> neighbors;
        neighbors.reserve(link_targets.size());
        while (!link_targets.empty()) {
            auto [dist, other] = link_targets.top();
            link_targets.pop();
            add_link(start_id, other);
            neighbors.push_back(other);
        }
        for (auto other : neighbors) {
            add_link(other, start_id);
        }
    }

    void add_link(vamana::idx_t src, vamana::idx_t dest) {
        auto& src_neighbors = nodes.at(src).neighbors;
        if (std::find(src_neighbors.begin(), src_neighbors.end(), dest) != src_neighbors.end()) {
            return;
        }
        if (src_neighbors.size() < Mmax) {
            src_neighbors.push_back(dest);
            return;
        }

        MaxHeap candidates;
        candidates.emplace(compute_distance(nodes.at(src).vector, nodes.at(dest).vector), dest);
        for (auto neighbor_id : src_neighbors) {
            candidates.emplace(compute_distance(nodes.at(src).vector, nodes.at(neighbor_id).vector), neighbor_id);
        }
        shrink_neighbor_list(candidates, Mmax);
        src_neighbors.clear();
        while (!candidates.empty()) {
            src_neighbors.push_back(candidates.top().second);
            candidates.pop();
        }
    }

    void greedy_search(vamana::idx_t start,
                       const Vector& query,
                       MaxHeap& results,
                       uint32_t search_width) const {
        struct Candidate {
            float dist;
            vamana::idx_t id;
        };
        auto cmp = [](const Candidate& a, const Candidate& b) { return a.dist > b.dist; };
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidates(cmp);

        std::unordered_set<vamana::idx_t> visited;
        const float start_dist = compute_distance(nodes.at(start).vector, query);
        candidates.push({start_dist, start});
        results.emplace(start_dist, start);
        visited.insert(start);

        const uint32_t beam_width = std::max<uint32_t>(search_width, 1);
        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();
            const float best_result_dist = results.empty() ? std::numeric_limits<float>::max() : results.top().first;
            if (current.dist > best_result_dist && results.size() >= beam_width) {
                break;
            }

            const auto& neighbors = nodes.at(current.id).neighbors;
            for (auto neighbor_id : neighbors) {
                if (visited.insert(neighbor_id).second) {
                    const float dist = compute_distance(nodes.at(neighbor_id).vector, query);
                    if (results.size() < beam_width || dist < results.top().first) {
                        candidates.push({dist, neighbor_id});
                        results.emplace(dist, neighbor_id);
                        if (results.size() > beam_width) {
                            results.pop();
                        }
                    }
                }
            }
        }
    }

    MaxHeap search_base_layer(vamana::idx_t start,
                              const Vector& query,
                              uint32_t ef) const {
        MaxHeap top_candidates;
        struct Candidate {
            float dist;
            vamana::idx_t id;
        };
        auto cmp = [](const Candidate& a, const Candidate& b) { return a.dist > b.dist; };
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidates(cmp);

        float lower_bound = compute_distance(nodes.at(start).vector, query);
        candidates.push({lower_bound, start});
        top_candidates.emplace(lower_bound, start);

        std::unordered_set<vamana::idx_t> visited;
        visited.insert(start);

        while (!candidates.empty()) {
            auto current = candidates.top();
            if (current.dist > lower_bound && top_candidates.size() >= ef) {
                break;
            }
            candidates.pop();

            for (auto neighbor_id : nodes.at(current.id).neighbors) {
                if (visited.insert(neighbor_id).second) {
                    const float dist = compute_distance(nodes.at(neighbor_id).vector, query);
                    if (top_candidates.size() < ef || dist < lower_bound) {
                        candidates.push({dist, neighbor_id});
                        top_candidates.emplace(dist, neighbor_id);
                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                        if (!top_candidates.empty()) {
                            lower_bound = top_candidates.top().first;
                        }
                    }
                }
            }
        }
        return top_candidates;
    }

    void greedy_update_nearest(vamana::idx_t& nearest,
                               float& nearest_dist,
                               const Vector& query) const {
        bool improved = true;
        while (improved) {
            improved = false;
            for (auto neighbor : nodes.at(nearest).neighbors) {
                const float dist = compute_distance(nodes.at(neighbor).vector, query);
                if (dist < nearest_dist) {
                    nearest_dist = dist;
                    nearest = neighbor;
                    improved = true;
                }
            }
        }
    }

    void shrink_neighbor_list(MaxHeap& results, uint32_t max_size) const {
        if (results.size() <= max_size) {
            return;
        }
        MinHeap inverted;
        while (!results.empty()) {
            inverted.emplace(results.top());
            results.pop();
        }
        std::vector<DistAndId> output;
        shrink_neighbor_list_robust(inverted, output, max_size);
        for (const auto& item : output) {
            results.emplace(item);
        }
    }

    void shrink_neighbor_list_robust(MinHeap& input,
                                     std::vector<DistAndId>& output,
                                     uint32_t max_size) const {
        while (!input.empty() && output.size() < max_size) {
            auto candidate = input.top();
            input.pop();
            bool keep = true;
            for (const auto& chosen : output) {
                const float dist = compute_distance(nodes.at(candidate.second).vector,
                                                     nodes.at(chosen.second).vector);
                if (alpha * dist <= candidate.first) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                output.push_back(candidate);
            }
        }
    }

    void mark_deleted(vamana::idx_t internal_id) {
        delete_list.insert(internal_id);
        if (delete_list.size() * 100 >= nodes.size() * kDeleteBatchThresholdPercent) {
            compact_graph();
        }
    }

    void compact_graph() {
        for (auto& [id, vertex] : nodes) {
            std::vector<DistAndId> candidate_dists;
            candidate_dists.reserve(vertex.neighbors.size());
            for (auto neighbor_id : vertex.neighbors) {
                if (!delete_list.contains(neighbor_id)) {
                    candidate_dists.emplace_back(
                        compute_distance(vertex.vector, nodes.at(neighbor_id).vector), neighbor_id);
                    continue;
                }
                for (auto nested : nodes.at(neighbor_id).neighbors) {
                    if (!delete_list.contains(nested)) {
                        candidate_dists.emplace_back(
                            compute_distance(vertex.vector, nodes.at(nested).vector), nested);
                    }
                }
            }
            MaxHeap heap(candidate_dists.begin(), candidate_dists.end());
            shrink_neighbor_list(heap, Mmax);
            vertex.neighbors.clear();
            while (!heap.empty()) {
                vertex.neighbors.push_back(heap.top().second);
                heap.pop();
            }
        }
        for (auto id : delete_list) {
            nodes.erase(id);
            reverse_id_map.erase(id);
        }
        delete_list.clear();
        if (!nodes.empty()) {
            entry_point = nodes.begin()->first;
        } else {
            entry_point = std::numeric_limits<vamana::idx_t>::max();
        }
    }

    ANNSResult search_single(const Vector& query,
                             uint32_t k,
                             uint32_t ef,
                             bool return_distances) const {
        ANNSResult result;
        if (nodes.empty()) {
            return result;
        }
        vamana::idx_t nearest = entry_point;
        float nearest_dist = compute_distance(nodes.at(nearest).vector, query);
        greedy_update_nearest(nearest, nearest_dist, query);

        const uint32_t effective_ef = std::max<uint32_t>({ef, ef_search, k});
        MaxHeap top = search_base_layer(nearest, query, effective_ef);
        while (top.size() > k) {
            top.pop();
        }
        result.ids.reserve(top.size());
        if (return_distances) {
            result.distances.reserve(top.size());
        }
        while (!top.empty()) {
            auto [dist, internal_id] = top.top();
            top.pop();
            if (delete_list.contains(internal_id)) {
                continue;
            }
            auto it = reverse_id_map.find(internal_id);
            if (it == reverse_id_map.end()) {
                continue;
            }
            result.ids.push_back(it->second);
            if (return_distances) {
                result.distances.push_back(dist);
            }
        }
        std::reverse(result.ids.begin(), result.ids.end());
        if (return_distances) {
            std::reverse(result.distances.begin(), result.distances.end());
        }
        result.actual_k = result.ids.size();
        return result;
    }

    DistanceMetric metric;
    uint32_t dimension;
    vamana::idx_t entry_point;
    vamana::idx_t next_internal_id;

    uint32_t M;
    uint32_t Mmax;
    uint32_t ef_construction;
    uint32_t ef_search;
    float alpha;

    std::unordered_map<vamana::idx_t, vamana::Vertex> nodes;
    std::unordered_set<vamana::idx_t> delete_list;
    std::unordered_map<VectorId, vamana::idx_t> id_map;
    std::unordered_map<vamana::idx_t, VectorId> reverse_id_map;
};

VamanaANNS::VamanaANNS() : impl_(std::make_unique<Impl>()), built_(false) {
    metrics_.reset();
}

VamanaANNS::~VamanaANNS() = default;

std::string VamanaANNS::version() const {
    return "1.0.0";
}

std::string VamanaANNS::description() const {
    return "Vamana graph-based ANN (DiskANN-inspired) with greedy search and robust pruning";
}

std::vector<DistanceMetric> VamanaANNS::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

bool VamanaANNS::supports_distance(DistanceMetric metric) const {
    auto supported = supported_distances();
    return std::find(supported.begin(), supported.end(), metric) != supported.end();
}

void VamanaANNS::fit(const std::vector<VectorEntry>& dataset,
                     const AlgorithmParams& params) {
    metrics_.reset();
    build_params_ = params;
    impl_->reset();

    auto build_start = std::chrono::high_resolution_clock::now();

    impl_->M = params.get<uint32_t>("M", 8);
    impl_->Mmax = params.get<uint32_t>("Mmax", 16);
    impl_->ef_construction = params.get<uint32_t>("efConstruction", 50);
    impl_->ef_search = params.get<uint32_t>("efSearch", 200);
    impl_->alpha = params.get<float>("alpha", kDefaultAlpha);
    impl_->metric = static_cast<DistanceMetric>(
        params.get<int>("metric", static_cast<int>(DistanceMetric::L2)));

    build_params_.set("M", impl_->M);
    build_params_.set("Mmax", impl_->Mmax);
    build_params_.set("efConstruction", impl_->ef_construction);
    build_params_.set("efSearch", impl_->ef_search);
    build_params_.set("alpha", impl_->alpha);
    build_params_.set("metric", static_cast<int>(impl_->metric));

    if (!supports_distance(impl_->metric)) {
        throw std::runtime_error("Vamana: unsupported distance metric");
    }

    if (dataset.empty()) {
        impl_->dimension = 0;
        build_params_.set("dimension", 0u);
        built_ = true;
        return;
    }

    impl_->dimension = static_cast<uint32_t>(dataset.front().second.size());
    build_params_.set("dimension", impl_->dimension);
    for (const auto& [id, vec] : dataset) {
        if (vec.size() != impl_->dimension) {
            throw std::runtime_error("Vamana: inconsistent vector dimensions");
        }
        impl_->insert_node(id, vec);
    }

    built_ = true;

    auto build_end = std::chrono::high_resolution_clock::now();
    metrics_.build_time_seconds = std::chrono::duration<double>(build_end - build_start).count();
    metrics_.index_size_bytes = get_memory_usage();
}

bool VamanaANNS::save(const std::string& path) const {
    if (!built_) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    const uint32_t version_tag = 1;
    out.write(reinterpret_cast<const char*>(&version_tag), sizeof(version_tag));
    out.write(reinterpret_cast<const char*>(&impl_->dimension), sizeof(impl_->dimension));
    uint32_t metric = static_cast<uint32_t>(impl_->metric);
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&impl_->M), sizeof(impl_->M));
    out.write(reinterpret_cast<const char*>(&impl_->Mmax), sizeof(impl_->Mmax));
    out.write(reinterpret_cast<const char*>(&impl_->ef_construction), sizeof(impl_->ef_construction));
    out.write(reinterpret_cast<const char*>(&impl_->ef_search), sizeof(impl_->ef_search));
    out.write(reinterpret_cast<const char*>(&impl_->alpha), sizeof(impl_->alpha));

    uint64_t node_count = impl_->nodes.size();
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
    for (const auto& [internal_id, vertex] : impl_->nodes) {
        out.write(reinterpret_cast<const char*>(&internal_id), sizeof(internal_id));
        uint32_t dim = static_cast<uint32_t>(vertex.vector.size());
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(vertex.vector.data()), dim * sizeof(float));
        uint32_t neighbor_count = static_cast<uint32_t>(vertex.neighbors.size());
        out.write(reinterpret_cast<const char*>(&neighbor_count), sizeof(neighbor_count));
        for (auto neighbor : vertex.neighbors) {
            out.write(reinterpret_cast<const char*>(&neighbor), sizeof(neighbor));
        }
    }

    uint64_t id_map_size = impl_->id_map.size();
    out.write(reinterpret_cast<const char*>(&id_map_size), sizeof(id_map_size));
    for (const auto& [external_id, internal_id] : impl_->id_map) {
        out.write(reinterpret_cast<const char*>(&external_id), sizeof(external_id));
        out.write(reinterpret_cast<const char*>(&internal_id), sizeof(internal_id));
    }

    return true;
}

bool VamanaANNS::load(const std::string& path) {
    metrics_.reset();
    impl_->reset();

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    uint32_t version_tag = 0;
    in.read(reinterpret_cast<char*>(&version_tag), sizeof(version_tag));
    if (version_tag != 1) {
        return false;
    }

    in.read(reinterpret_cast<char*>(&impl_->dimension), sizeof(impl_->dimension));
    uint32_t metric = 0;
    in.read(reinterpret_cast<char*>(&metric), sizeof(metric));
    impl_->metric = static_cast<DistanceMetric>(metric);
    in.read(reinterpret_cast<char*>(&impl_->M), sizeof(impl_->M));
    in.read(reinterpret_cast<char*>(&impl_->Mmax), sizeof(impl_->Mmax));
    in.read(reinterpret_cast<char*>(&impl_->ef_construction), sizeof(impl_->ef_construction));
    in.read(reinterpret_cast<char*>(&impl_->ef_search), sizeof(impl_->ef_search));
    in.read(reinterpret_cast<char*>(&impl_->alpha), sizeof(impl_->alpha));

    uint64_t node_count = 0;
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
    for (uint64_t i = 0; i < node_count; ++i) {
        vamana::idx_t internal_id = 0;
        in.read(reinterpret_cast<char*>(&internal_id), sizeof(internal_id));
        uint32_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        Vector vec(dim);
        in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        vamana::Vertex vertex(internal_id, std::move(vec));
        uint32_t neighbor_count = 0;
        in.read(reinterpret_cast<char*>(&neighbor_count), sizeof(neighbor_count));
        vertex.neighbors.resize(neighbor_count);
        in.read(reinterpret_cast<char*>(vertex.neighbors.data()), neighbor_count * sizeof(vamana::idx_t));
        impl_->nodes.emplace(internal_id, std::move(vertex));
        impl_->next_internal_id = std::max(impl_->next_internal_id, internal_id + 1);
    }

    uint64_t id_map_size = 0;
    in.read(reinterpret_cast<char*>(&id_map_size), sizeof(id_map_size));
    for (uint64_t i = 0; i < id_map_size; ++i) {
        VectorId external_id = 0;
        vamana::idx_t internal_id = 0;
        in.read(reinterpret_cast<char*>(&external_id), sizeof(external_id));
        in.read(reinterpret_cast<char*>(&internal_id), sizeof(internal_id));
        impl_->id_map.emplace(external_id, internal_id);
        impl_->reverse_id_map.emplace(internal_id, external_id);
    }

    if (!impl_->nodes.empty()) {
        impl_->entry_point = impl_->nodes.begin()->first;
    }

    built_ = true;
    return true;
}

ANNSResult VamanaANNS::query(const Vector& query_vector,
                             const QueryConfig& config) const {
    if (!built_ || impl_->dimension == 0) {
        return {};
    }
    if (query_vector.size() != impl_->dimension) {
        throw std::runtime_error("Vamana: query dimension mismatch");
    }

    const uint32_t ef_override = config.algorithm_params.get<uint32_t>(
        "efSearch", impl_->ef_search);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = impl_->search_single(query_vector,
                                       config.k,
                                       ef_override,
                                       config.return_distances);
    auto end = std::chrono::high_resolution_clock::now();
    metrics_.search_time_seconds += std::chrono::duration<double>(end - start).count();
    metrics_.distance_computations += static_cast<size_t>(result.actual_k) * impl_->dimension;
    return result;
}

std::vector<ANNSResult> VamanaANNS::batch_query(
    const std::vector<Vector>& query_vectors,
    const QueryConfig& config) const {
    if (!built_ || impl_->dimension == 0) {
        return {};
    }

    std::vector<ANNSResult> results;
    results.reserve(query_vectors.size());

    const uint32_t ef_override = config.algorithm_params.get<uint32_t>(
        "efSearch", impl_->ef_search);

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& query : query_vectors) {
        if (query.size() != impl_->dimension) {
            throw std::runtime_error("Vamana: query dimension mismatch");
        }
        results.push_back(impl_->search_single(query,
                                               config.k,
                                               ef_override,
                                               config.return_distances));
    }
    auto end = std::chrono::high_resolution_clock::now();
    metrics_.search_time_seconds += std::chrono::duration<double>(end - start).count();
    size_t total_neighbors = 0;
    for (const auto& res : results) {
        total_neighbors += res.actual_k;
    }
    metrics_.distance_computations += total_neighbors * impl_->dimension;
    return results;
}

void VamanaANNS::add_vector(const VectorEntry& entry) {
    if (!built_) {
        throw std::runtime_error("Vamana: index not built");
    }
    if (entry.second.size() != impl_->dimension) {
        throw std::runtime_error("Vamana: vector dimension mismatch");
    }
    impl_->insert_node(entry.first, entry.second);
}

void VamanaANNS::add_vectors(const std::vector<VectorEntry>& entries) {
    for (const auto& entry : entries) {
        add_vector(entry);
    }
}

void VamanaANNS::remove_vector(VectorId id) {
    auto it = impl_->id_map.find(id);
    if (it == impl_->id_map.end()) {
        return;
    }
    impl_->mark_deleted(it->second);
    impl_->id_map.erase(it);
}

void VamanaANNS::remove_vectors(const std::vector<VectorId>& ids) {
    for (auto id : ids) {
        remove_vector(id);
    }
}

size_t VamanaANNS::get_index_size() const {
    return impl_->nodes.size() - impl_->delete_list.size();
}

size_t VamanaANNS::get_memory_usage() const {
    size_t total = 0;
    for (const auto& [_, vertex] : impl_->nodes) {
        total += vertex.vector.size() * sizeof(float);
        total += vertex.neighbors.size() * sizeof(vamana::idx_t);
    }
    total += impl_->id_map.size() * (sizeof(VectorId) + sizeof(vamana::idx_t));
    return total;
}

std::unordered_map<std::string, std::string> VamanaANNS::get_build_params() const {
    return build_params_.params;
}

ANNSMetrics VamanaANNS::get_metrics() const {
    return metrics_;
}

bool VamanaANNS::validate_params(const AlgorithmParams& params) const {
    const auto M = params.get<uint32_t>("M", 8);
    const auto Mmax = params.get<uint32_t>("Mmax", 16);
    const auto efC = params.get<uint32_t>("efConstruction", 50);
    const auto efS = params.get<uint32_t>("efSearch", 200);
    const auto alpha = params.get<float>("alpha", kDefaultAlpha);
    const auto metric = static_cast<DistanceMetric>(
        params.get<int>("metric", static_cast<int>(DistanceMetric::L2)));
    return M > 0 && Mmax >= M && efC > 0 && efS > 0 && alpha > 0.0f && supports_distance(metric);
}

AlgorithmParams VamanaANNS::get_default_params() const {
    AlgorithmParams defaults;
    defaults.set("M", 8u);
    defaults.set("Mmax", 16u);
    defaults.set("efConstruction", 50u);
    defaults.set("efSearch", 200u);
    defaults.set("alpha", kDefaultAlpha);
    defaults.set("metric", static_cast<int>(DistanceMetric::L2));
    return defaults;
}

QueryConfig VamanaANNS::get_default_query_config() const {
    QueryConfig config;
    config.k = 10;
    config.return_distances = true;
    config.algorithm_params = AlgorithmParams{};
    return config;
}

std::unique_ptr<ANNSAlgorithm> VamanaANNSFactory::create() const {
    return std::make_unique<VamanaANNS>();
}

std::string VamanaANNSFactory::algorithm_description() const {
    return "Vamana graph ANN with greedy search and robust pruning";
}

std::vector<DistanceMetric> VamanaANNSFactory::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

AlgorithmParams VamanaANNSFactory::default_build_params() const {
    return VamanaANNS().get_default_params();
}

QueryConfig VamanaANNSFactory::default_query_config() const {
    return VamanaANNS().get_default_query_config();
}

}  // namespace anns
}  // namespace sage_vdb
