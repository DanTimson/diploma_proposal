// D-007: explicit SIMD reduction; no fast-math, one CPU thread.
// Exhaustive squared-L2 search; optional one-per-group constraint.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
extern "C" int exact_groups_simd(const float* x, const float* q, const int64_t* ids,
 const int64_t* groups, int64_t count, int64_t n, int d, int k, int grouped, int64_t* out) {
 if(n<=0 || count<0 || d<=0 || k<=0) return -1;
 struct Hit { double distance; int64_t id; };
 auto less=[](const Hit&a,const Hit&b) {return a.distance<b.distance || (a.distance==b.distance && a.id<b.id);};
 std::vector<Hit> best(grouped?n:0,{std::numeric_limits<double>::infinity(),-1}),hits;
 hits.reserve(count);
 for(int64_t j=0;j<count;++j) {
  auto id=ids[j];if(id<0 || id>=n)return -2;
  double value=0;const float* row=x+id*d;
  #pragma omp simd reduction(+:value)
  for(int z=0;z<d;++z){double delta=double(row[z])-q[z];value+=delta*delta;}
  Hit h{value,id};
  if(grouped){auto g=groups[id];if(g<0 || g>=n)return -3;if(less(h,best[g]))best[g]=h;}
  else hits.push_back(h);
 }
 if(grouped)for(auto h:best)if(h.id>=0)hits.push_back(h);
 auto wanted=std::min(size_t(k),hits.size());
 std::partial_sort(hits.begin(),hits.begin()+wanted,hits.end(),less);
 for(size_t i=0;i<wanted;++i)out[i]=hits[i].id;
 return int(wanted);
}
