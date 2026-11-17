// D-008 diagnostic, not a published HNSW/ACORN/Diverse Search reproduction.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>
using Hit=std::pair<double,int64_t>;
extern "C" int retention(const float* x,const float* q,const int64_t* edges,
 const int64_t* groups,const uint8_t* allowed,int64_t n,int d,int width,
 const int64_t* starts,int ns,int budget,int beam,int mode,int reserve,
 int k,int64_t* out,int64_t* stats) {
 if(n<=0||d<=0||width<=0||ns<=0||budget<ns||beam<=0||mode<0||mode>3||reserve||k<=0)return -1;
 std::fill(stats,stats+8,0);
 std::vector<uint8_t> seen(n,0),expanded_group(n,0);
 std::vector<Hit> best(n,{std::numeric_limits<double>::infinity(),-1});
 std::set<Hit> pending,eligible; // eligible is an auxiliary index of shared slots.
 auto discover=[&](int64_t id){
   if(id<0||id>=n||seen[id]||stats[0]>=budget)return;
   seen[id]=1;++stats[0];double distance=0;
   #pragma omp simd reduction(+:distance)
   for(int j=0;j<d;++j){double v=double(x[id*d+j])-q[j];distance+=v*v;}
   Hit h{distance,id};auto g=groups[id];
   if(allowed[id])++stats[2];else ++stats[5];
   // after modes keep the best unfiltered member and filter only at output.
   if(allowed[id]||mode>=2){
     if(best[g].second<0)++stats[3];else if(h<best[g])++stats[4];
     if(h<best[g])best[g]=h;
   }
   pending.insert(h);if((mode%2)&&allowed[id])eligible.insert(h);
   if(pending.size()>size_t(beam)){
     auto worst=std::prev(pending.end());eligible.erase(*worst);pending.erase(worst);
   }
   stats[7]=std::max<int64_t>(stats[7],pending.size());
 };
 for(int j=0;j<ns;++j)discover(starts[j]);
 while(!pending.empty()&&stats[0]<budget){
   bool prefer=(mode%2)&&stats[1]%2==1&&!eligible.empty();
   Hit h=prefer?*eligible.begin():*pending.begin();
   pending.erase(h);eligible.erase(h);
   ++stats[1];auto g=groups[h.second];stats[6]+=expanded_group[g]!=0;expanded_group[g]=1;
   for(int j=0;j<width&&stats[0]<budget;++j)discover(edges[h.second*width+j]);
 }
 std::vector<Hit> answers;
 for(auto h:best)if(h.second>=0&&allowed[h.second])answers.push_back(h);
 auto count=std::min<size_t>(k,answers.size());
 std::partial_sort(answers.begin(),answers.begin()+count,answers.end());
 for(size_t j=0;j<count;++j)out[j]=answers[j].second;
 return int(count);
}
