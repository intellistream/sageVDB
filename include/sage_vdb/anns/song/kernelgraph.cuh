#pragma once

#include "sage_vdb/anns/song/config.hpp"
#include "sage_vdb/anns/song/data.hpp"
#include "sage_vdb/anns/song/bin_heap.hpp"
#include "sage_vdb/anns/song/smmh2.hpp"
#include "sage_vdb/anns/song/fixhash.hpp"
#include "sage_vdb/anns/song/bloomfilter.hpp"
#include "sage_vdb/anns/song/blocked_bloomfilter.hpp"
#include "sage_vdb/anns/song/cuckoofilter.hpp"
#include "sage_vdb/anns/song/warp_astar_accelerator.cuh"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace song_kernel {

class GraphWrapper {
public:
    virtual ~GraphWrapper() = default;
    virtual void add_vertex(idx_t vertex_id, std::vector<std::pair<int, value_t>>& point) = 0;
    virtual void search_top_k(const std::vector<std::pair<int, value_t>>& query, int k,
                              std::vector<idx_t>& result) = 0;
    virtual void search_top_k_batch(const std::vector<std::vector<std::pair<int, value_t>>>& queries,
                                    int k, std::vector<std::vector<idx_t>>& results) {}
    virtual void dump(std::string file = "song.graph") = 0;
    virtual void load(std::string file = "song.graph") = 0;
};

template <const int dist_type>
class KernelFixedDegreeGraph : public GraphWrapper {
private:
    const int degree = 15;
    const int flexible_degree = degree * 2 + 1;
    const int vertex_offset_shift = 5;
    std::vector<idx_t> edges;
    std::vector<dist_t> edge_dist;
    Data* data;
    std::mt19937_64 rand_gen = std::mt19937_64(114514);

    template <class T>
    dist_t distance(idx_t a, T& b) {
        if constexpr (dist_type == 0) {
            return data->l2_distance(a, b);
        } else if constexpr (dist_type == 1) {
            return data->negative_inner_prod_distance(a, b);
        } else {
            return data->negative_cosine_distance(a, b);
        }
    }

    template <class T>
    dist_t pair_distance(idx_t a, T& b) {
        return distance(a, b);
    }

    void compute_distance(size_t offset, std::vector<dist_t>& dists) {
        dists.resize(edges[offset]);
        auto deg = edges[offset];
        for (int i = 0; i < deg; ++i) {
            dists[i] = distance(offset >> vertex_offset_shift, edges[offset + i + 1]);
        }
    }

    void qsort(size_t l, size_t r) {
        auto mid = (l + r) >> 1;
        int i = l, j = r;
        auto pivot = edge_dist[mid];
        do {
            while (edge_dist[i] < pivot) ++i;
            while (pivot < edge_dist[j]) --j;
            if (i <= j) {
                std::swap(edge_dist[i], edge_dist[j]);
                std::swap(edges[i], edges[j]);
                ++i;
                --j;
            }
        } while (i <= j);
        if (i < static_cast<int>(r)) qsort(i, r);
        if (static_cast<int>(l) < j) qsort(l, j);
    }

    void rank_edges(size_t offset) {
        std::vector<dist_t> dists;
        compute_distance(offset, dists);
        for (size_t i = 0; i < dists.size(); ++i) {
            edge_dist[offset + i + 1] = dists[i];
        }
        qsort(offset + 1, offset + dists.size());
    }

    void rank_and_switch_ordered(idx_t v_id, idx_t u_id) {
        auto curr_dist = pair_distance(v_id, u_id);
        auto offset = v_id << vertex_offset_shift;
        if (curr_dist >= edge_dist[offset + edges[offset]]) {
            return;
        }
        edges[offset + edges[offset]] = u_id;
        edge_dist[offset + edges[offset]] = curr_dist;
        for (size_t i = offset + edges[offset] - 1; i > offset; --i) {
            if (edge_dist[i] > edge_dist[i + 1]) {
                std::swap(edges[i], edges[i + 1]);
                std::swap(edge_dist[i], edge_dist[i + 1]);
            } else {
                break;
            }
        }
    }

    void rank_and_switch(idx_t v_id, idx_t u_id) {
        rank_and_switch_ordered(v_id, u_id);
    }

    void add_edge(idx_t v_id, idx_t u_id) {
        auto offset = v_id << vertex_offset_shift;
        if (edges[offset] < flexible_degree) {
            ++edges[offset];
            edges[offset + edges[offset]] = u_id;
            if (edges[offset] == flexible_degree) {
                rank_edges(offset);
            }
        } else {
            rank_and_switch(v_id, u_id);
        }
    }

public:
    long long total_explore_cnt = 0;
    int total_explore_times = 0;

    explicit KernelFixedDegreeGraph(Data* data) : data(data) {
        auto num_vertices = data->max_vertices();
        edges = std::vector<idx_t>(num_vertices << vertex_offset_shift);
        edge_dist = std::vector<dist_t>(num_vertices << vertex_offset_shift);
    }

    void add_vertex(idx_t vertex_id, std::vector<std::pair<int, value_t>>& point) override {
        std::vector<idx_t> neighbor;
        search_top_k(point, degree * 10, neighbor);
        int num_neighbors = degree < static_cast<int>(neighbor.size()) ? degree : neighbor.size();
        auto offset = vertex_id << vertex_offset_shift;
        edges[offset] = num_neighbors;
        for (int i = 0; i < static_cast<int>(neighbor.size()) && i < degree; ++i) {
            edges[offset + i + 1] = neighbor[i];
        }
        rank_edges(offset);
        for (int i = 0; i < static_cast<int>(neighbor.size()) && i < degree; ++i) {
            add_edge(neighbor[i], vertex_id);
        }
    }

    void add_vertex_new(idx_t vertex_id, std::vector<std::pair<int, value_t>>& point) {
        std::vector<std::vector<idx_t>> neighbor;
        std::vector<std::vector<std::pair<int, value_t>>> points(1, point);
        search_top_k_batch(points, degree * 10, neighbor);
        int num_neighbors = degree < static_cast<int>(neighbor[0].size()) ? degree : neighbor[0].size();
        auto offset = vertex_id << vertex_offset_shift;
        edges[offset] = num_neighbors;
        for (int i = 0; i < static_cast<int>(neighbor[0].size()) && i < degree; ++i) {
            edges[offset + i + 1] = neighbor[0][i];
        }
        rank_edges(offset);
        for (int i = 0; i < static_cast<int>(neighbor[0].size()) && i < degree; ++i) {
            add_edge(neighbor[0][i], vertex_id);
        }
    }

    void search_top_k(const std::vector<std::pair<int, value_t>>& query, int k,
                      std::vector<idx_t>& result) override {
        WarpAStarAccelerator::astar_multi_start_search(query, k, result, data->get(0), edges.data(),
                                                       vertex_offset_shift, data->max_vertices(),
                                                       data->get_dim(), dist_type);
    }

    void dump(std::string file = "song.graph") override {
        FILE* fp = std::fopen(file.c_str(), "wb");
        if (!fp) {
            return;
        }
        auto num_vertices = data->max_vertices();
        std::fwrite(&edges[0], sizeof(edges[0]) * (num_vertices << vertex_offset_shift), 1, fp);
        std::fclose(fp);
    }

    void load(std::string file = "song.graph") override {
        FILE* fp = std::fopen(file.c_str(), "rb");
        if (!fp) {
            return;
        }
        auto num_vertices = data->max_vertices();
        std::fread(&edges[0], sizeof(edges[0]) * (num_vertices << vertex_offset_shift), 1, fp);
        std::fclose(fp);
    }

    void search_top_k_batch(const std::vector<std::vector<std::pair<int, value_t>>>& queries, int k,
                            std::vector<std::vector<idx_t>>& results) override {
        WarpAStarAccelerator::astar_multi_start_search_batch(queries, k, results, data->get(0),
                                                             edges.data(), vertex_offset_shift,
                                                             data->max_vertices(), data->get_dim(),
                                                             dist_type);
    }
};

} // namespace song_kernel
