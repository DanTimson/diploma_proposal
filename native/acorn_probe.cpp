// Separate process: never mix the upstream FAISS fork with the pip FAISS library.
#include <faiss/IndexACORN.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>
using Clock=std::chrono::steady_clock;
template<class T> std::vector<T> read_raw(const char* path, size_t count) {
    std::ifstream f(path,std::ios::binary); std::vector<T> v(count);
    f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));
    if(!f || f.peek()!=EOF) throw std::runtime_error("Bad raw file length"); return v;
}
std::vector<float> read_vectors(const char* path,uint32_t& n,uint32_t& d) {
    std::ifstream f(path,std::ios::binary); f.read(reinterpret_cast<char*>(&n),4);f.read(reinterpret_cast<char*>(&d),4);
    if(!f || !n || !d || d>100000) throw std::runtime_error("Bad vector header");
    std::vector<float> v(size_t(n)*d);f.read(reinterpret_cast<char*>(v.data()),v.size()*4);
    if(!f || f.peek()!=EOF) throw std::runtime_error("Bad vector length");return v;
}
int main(int argc,char** argv) {
  try {
    if(argc!=13) throw std::runtime_error("base query masks groups metadata output k pool ef gamma seed repeats");
    uint32_t n,d,nq,dq; auto x=read_vectors(argv[1],n,d);auto q=read_vectors(argv[2],nq,dq);
    if(d!=dq) throw std::runtime_error("Dimension mismatch");
    auto masks=read_raw<char>(argv[3],size_t(n)*nq);auto groups=read_raw<int64_t>(argv[4],n);
    auto metadata=read_raw<int>(argv[5],n);
    int k=std::stoi(argv[7]),pool=std::stoi(argv[8]),ef=std::stoi(argv[9]),gamma=std::stoi(argv[10]);
    int seed=std::stoi(argv[11]),reps=std::stoi(argv[12]);
    if(k<1 || pool<k || pool>int(n) || ef<1 || gamma<1 || reps<1) throw std::runtime_error("Invalid budgets");
    for(auto g:groups) if(g<0 || g>=n) throw std::runtime_error("Bad group");
    omp_set_num_threads(1);
    auto t=Clock::now(); faiss::IndexACORNFlat index(d,32,gamma,metadata,gamma==1?64:32);
    index.acorn.rng=faiss::RandomGenerator(seed);index.acorn.efConstruction=200;
    index.add(n,x.data());
    double build=std::chrono::duration<double>(Clock::now()-t).count();
    index.acorn.efSearch=std::max(ef,pool);
    std::vector<int64_t> output(size_t(nq)*k,-1), rejected(nq,0);std::vector<double> timings(nq);
    std::vector<faiss::idx_t> ids(pool);std::vector<float> distances(pool);
    for(uint32_t qi=0;qi<nq;++qi) {
      std::vector<double> samples;
      for(int rep=0;rep<=reps;++rep) {
        auto start=Clock::now();
        index.search(1,q.data()+size_t(qi)*d,pool,distances.data(),ids.data(),masks.data()+size_t(qi)*n);
        std::vector<std::pair<double,int64_t>> sorted; int64_t filtered=0;
        for(auto id:ids) {
          if(id==-1) continue;
          if(id<0 || id>=n) throw std::runtime_error("Invalid result ID");
          // Upstream seeds the result heap without consulting filter_map.
          // Recheck authoritatively, and expose the rejected count in receipts.
          if(!masks[size_t(qi)*n+id]) {++filtered;continue;}
          double dist=0;for(uint32_t j=0;j<d;++j) {double v=double(x[size_t(id)*d+j])-q[size_t(qi)*d+j];dist+=v*v;}
          sorted.emplace_back(dist,id);
        }
        std::sort(sorted.begin(),sorted.end());std::set<int64_t> seen;int count=0;
        std::fill(output.begin()+size_t(qi)*k,output.begin()+size_t(qi+1)*k,-1);
        for(auto item:sorted) if(seen.insert(groups[item.second]).second) {
          output[size_t(qi)*k+count++]=item.second;if(count==k) break;
        }
        double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        if(rep) samples.push_back(ms);
        if(rep==reps) rejected[qi]=filtered;
      }
      std::sort(samples.begin(),samples.end());
      timings[qi]=(samples[(samples.size()-1)/2]+samples[samples.size()/2])/2;
    }
    std::ofstream f(argv[6],std::ios::binary);uint32_t ku=k;
    f.write(reinterpret_cast<char*>(&nq),4);f.write(reinterpret_cast<char*>(&ku),4);
    f.write(reinterpret_cast<char*>(output.data()),output.size()*8);
    f.write(reinterpret_cast<char*>(timings.data()),timings.size()*8);
    f.write(reinterpret_cast<char*>(rejected.data()),rejected.size()*8);
    if(!f) throw std::runtime_error("Write failed");
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    std::cout<<"D004_BUILD_SECONDS "<<build<<"\nD004_PEAK_RSS_KIB "<<usage.ru_maxrss<<"\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
