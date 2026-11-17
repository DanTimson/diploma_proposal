// D-006 diagnostic traversal. NOT an implementation of HNSW/ACORN/DiverseSearch.
// All nodes can route; eligibility constrains answers only. Distance budget is strict.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>
using Hit = std::pair<double,int64_t>;
extern "C" int explore_groups(const float* x, const float* q, const int64_t* edges,
 const int64_t* groups, const uint8_t* allowed, int64_t n, int d, int width,
 const int64_t* starts, int ns, int budget, int beam, int cap, int reserve,
 int k, int64_t* out, int64_t* stats) {
 if(n<=0 || d<=0 || width<=0 || ns<=0 || budget<ns || beam<=reserve ||
    reserve<0 || cap<0 || k<=0) return -1;
 std::fill(stats,stats+8,0);
 std::vector<uint8_t> seen(n,0), expanded_group(n,0);
 std::vector<Hit> best(n,{std::numeric_limits<double>::infinity(),-1});
 std::set<Hit> frontier, bridges;
 std::vector<int> counts(n,0);
 const size_t main_limit=beam-reserve;
 auto bridge=[&](Hit h) {
   if(!reserve)return;
   bridges.insert(h);
   if(bridges.size()>size_t(reserve))bridges.erase(std::prev(bridges.end()));
 };
 auto admit=[&](Hit h) {
   int64_t g=groups[h.second];
   if(cap && counts[g]>=cap) {
     auto worst=frontier.end();
     for(auto it=frontier.begin();it!=frontier.end();++it)
       if(groups[it->second]==g)worst=it;
     if(worst==frontier.end() || ! (h<*worst)){bridge(h);return;}
     bridge(*worst);frontier.erase(worst);--counts[g];
   }
   frontier.insert(h);++counts[g];
   if(frontier.size()>main_limit){auto it=std::prev(frontier.end());
     --counts[groups[it->second]];bridge(*it);frontier.erase(it);}
   stats[7]=std::max<int64_t>(stats[7],frontier.size()+bridges.size());
 };
 auto discover=[&](int64_t id) {
   if(id<0 || id>=n || seen[id] || stats[0]>=budget)return;
   seen[id]=1;++stats[0];double dist=0;
   for(int j=0;j<d;++j){double v=double(x[id*d+j])-q[j];dist+=v*v;}
   Hit h{dist,id};auto g=groups[id];
   if(allowed[id]){++stats[2];
     if(best[g].second<0)++stats[3];
     else if(h<best[g])++stats[4];
     if(h<best[g])best[g]=h;
   }else ++stats[5];
   admit(h);
   stats[7]=std::max<int64_t>(stats[7],frontier.size()+bridges.size());
 };
 for(int j=0;j<ns;++j)discover(starts[j]);
 while((!frontier.empty() || !bridges.empty()) && stats[0]<budget){
   bool use_bridge=!bridges.empty() && (frontier.empty() || stats[1]%4==3);
   auto &queue=use_bridge?bridges:frontier;
   auto h=*queue.begin();queue.erase(queue.begin());
   if(!use_bridge)--counts[groups[h.second]];
   ++stats[1];auto g=groups[h.second];
   stats[6]+=expanded_group[g]!=0;expanded_group[g]=1;
   for(int j=0;j<width && stats[0]<budget;++j)discover(edges[h.second*width+j]);
 }
 std::vector<Hit> answers;
 for(auto h:best)if(h.second>=0)answers.push_back(h);
 auto count=std::min<size_t>(k,answers.size());
 std::partial_sort(answers.begin(),answers.begin()+count,answers.end());
 for(size_t j=0;j<count;++j)out[j]=answers[j].second;
 return int(count);
}
