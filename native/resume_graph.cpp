// Diagnostic traversal of an exported HNSW level-0 graph, NOT FAISS search.
// All discovered nodes remain routable; predicates and group caps affect outputs.
#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>
#include <utility>
#include <new>
using Node = std::pair<double, int64_t>;
struct State {
    const float *x; const int64_t *edges; int64_t n, width; int d;
    std::vector<float> q;
    std::vector<unsigned char> seen;
    std::vector<Node> discovered;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> frontier;
    int64_t expanded = 0, peak = 0;
    void visit(int64_t id) {
        if (id < 0 || seen[id]) return;
        seen[id] = 1; double dist = 0;
        for (int j=0; j<d; ++j) { double v=double(x[id*d+j])-q[j]; dist+=v*v; }
        Node p{dist,id}; discovered.push_back(p); frontier.push(p);
        peak=std::max(peak, int64_t(frontier.size()));
    }
    State(const float* x_, const float* q_, const int64_t* e, int64_t n_, int d_, int64_t w, int64_t entry)
        : x(x_), edges(e), n(n_), width(w), d(d_), q(q_, q_+d_), seen(n_,0) { visit(entry); }
};
extern "C" {
void* rg_create(const float* x, const float* q, const int64_t* edges, int64_t n, int d, int64_t width, int64_t entry) {
    try { return new State(x,q,edges,n,d,width,entry); } catch (...) { return nullptr; }
}
void rg_free(void* p) { delete static_cast<State*>(p); }
int rg_advance(void* p, int64_t target) {
    try {
        auto& s=*static_cast<State*>(p);
        if (target<s.expanded) return -2;
        while (s.expanded<target && !s.frontier.empty()) {
            auto id=s.frontier.top().second; s.frontier.pop(); ++s.expanded;
            for (int64_t j=0;j<s.width;++j) s.visit(s.edges[id*s.width+j]);
        }
        return 0;
    } catch (...) { return -1; }
}
// mode=0: shortlist cap precedes eligibility; mode=1: eligibility precedes cap.
int rg_select(void* p, const int64_t* groups, const uint8_t* mask, int k,
              int pool, int cap, int mode, int64_t* out) {
    try {
        auto& s=*static_cast<State*>(p); auto nodes=s.discovered;
        std::sort(nodes.begin(),nodes.end());
        std::vector<int> counts(s.n,0); std::vector<uint8_t> used(s.n,0);
        int accepted=0, count=0;
        for (auto item:nodes) {
            auto id=item.second, g=groups[id];
            if (mode && !mask[id]) continue;
            if (cap && counts[g]>=cap) continue;
            ++counts[g]; ++accepted;
            if (mask[id] && !used[g]) { out[count++]=id; used[g]=1; if(count==k) break; }
            if (accepted==pool) break;
        }
        return count;
    } catch (...) { return -1; }
}
void rg_stats(void* p, int64_t* out) {
    auto& s=*static_cast<State*>(p);
    out[0]=s.expanded; out[1]=s.discovered.size(); out[2]=s.frontier.size(); out[3]=s.peak;
}
}
