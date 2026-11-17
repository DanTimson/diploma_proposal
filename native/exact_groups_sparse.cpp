// Same exact objective/ABI as exact_groups.cpp; aggregation visits eligible groups only.
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>
extern "C" int exact_groups(const float* x, const float* q, const int64_t* ids,
 const int64_t* groups, int64_t count, int64_t n, int d, int k, int grouped, int64_t* out) {
 if(n<=0 || count<0 || d<=0 || k<=0) return -1;
 struct Hit { double distance; int64_t id; };
 auto less=[](const Hit&a,const Hit&b){return a.distance<b.distance || (a.distance==b.distance && a.id<b.id);};
 try {
  std::unordered_map<int64_t,Hit> best;
  if(grouped)best.reserve(std::min<int64_t>(count,4096));
  std::vector<Hit> hits;if(!grouped)hits.reserve(count);
  for(int64_t j=0;j<count;++j) {
   auto id=ids[j];if(id<0 || id>=n)return -2;
   double value=0;const float* row=x+id*d;
   for(int z=0;z<d;++z){double delta=double(row[z])-q[z];value+=delta*delta;}
   Hit h{value,id};
   if(grouped){auto g=groups[id];if(g<0 || g>=n)return -3;
    auto it=best.find(g);if(it==best.end())best.emplace(g,h);else if(less(h,it->second))it->second=h;
   } else hits.push_back(h);
  }
  if(grouped){hits.reserve(best.size());for(const auto& pair:best)hits.push_back(pair.second);}
  auto wanted=std::min(size_t(k),hits.size());
  std::partial_sort(hits.begin(),hits.begin()+wanted,hits.end(),less);
  for(size_t i=0;i<wanted;++i)out[i]=hits[i].id;
  return int(wanted);
 } catch (...) {return -4;}
}
