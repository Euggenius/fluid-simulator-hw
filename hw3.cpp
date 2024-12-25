#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <vector>
#include <array>
#include <memory>
using namespace std;

using FLOAT_t = float;
using DOUBLE_t = double;

template<int N, int K>
struct Fixed {
    static_assert(N > K, "N should be greater than K");
    static_assert(N <= 64, "Too large N");

    using underlying_type = std::conditional_t<(N <= 8), int8_t,
                           std::conditional_t<(N <= 16), int16_t,
                           std::conditional_t<(N <= 32), int32_t, int64_t>>>;

    underlying_type v;

    constexpr Fixed() : v(0) {}
    constexpr Fixed(int iv) : v(static_cast<underlying_type>(iv << K)) {}
    constexpr Fixed(float fv) : v(static_cast<underlying_type>(fv * (1 << K))) {}
    constexpr Fixed(double dv) : v(static_cast<underlying_type>(dv * (1 << K))) {}

    static constexpr Fixed from_raw(underlying_type x) {
        Fixed ret;
        ret.v = x;
        return ret;
    }

    explicit operator double() const {
        return static_cast<double>(v) / (1 << K);
    }

    bool operator==(const Fixed &other) const { return v == other.v; }
    bool operator!=(const Fixed &other) const { return v != other.v; }
    bool operator<(const Fixed &other) const { return v < other.v; }
    bool operator>(const Fixed &other) const { return v > other.v; }
    bool operator<=(const Fixed &other) const { return v <= other.v; }
    bool operator>=(const Fixed &other) const { return v >= other.v; }

    friend Fixed operator+(Fixed a, Fixed b) {
        return Fixed::from_raw(a.v + b.v);
    }
    friend Fixed operator-(Fixed a, Fixed b) {
        return Fixed::from_raw(a.v - b.v);
    }
    friend Fixed operator-(Fixed x) {
        return Fixed::from_raw(-x.v);
    }
    friend Fixed operator*(Fixed a, Fixed b) {
        __int128 temp = static_cast<__int128>(a.v) * b.v;
        return Fixed::from_raw(static_cast<underlying_type>(temp >> K));
    }
    friend Fixed operator/(Fixed a, Fixed b) {
        __int128 temp = (static_cast<__int128>(a.v) << K) / b.v;
        return Fixed::from_raw(static_cast<underlying_type>(temp));
    }

    Fixed &operator+=(Fixed b) { *this = *this + b; return *this; }
    Fixed &operator-=(Fixed b) { *this = *this - b; return *this; }
    Fixed &operator*=(Fixed b) { *this = *this * b; return *this; }
    Fixed &operator/=(Fixed b) { *this = *this / b; return *this; }

    friend ostream &operator<<(ostream &out, Fixed x) {
        double val = static_cast<double>(x);
        return out << fixed << setprecision(2) << val;
    }
};

template<int N, int K>
struct FastFixed {
    using underlying_type = std::conditional_t<(N <= 8), int_fast8_t,
                           std::conditional_t<(N <= 16), int_fast16_t,
                           std::conditional_t<(N <= 32), int_fast32_t, int_fast64_t>>>;

    static_assert(sizeof(underlying_type) * 8 >= static_cast<size_t>(N), "No suitable fast type");

    underlying_type v;

    constexpr FastFixed() : v(0) {}
    constexpr FastFixed(int iv) : v(static_cast<underlying_type>(iv << K)) {}
    constexpr FastFixed(float fv) : v(static_cast<underlying_type>(fv * (1 << K))) {}
    constexpr FastFixed(double dv) : v(static_cast<underlying_type>(dv * (1 << K))) {}

    static constexpr FastFixed from_raw(underlying_type x) {
        FastFixed ret;
        ret.v = x;
        return ret;
    }

    explicit operator double() const {
        return static_cast<double>(v) / (1 << K);
    }

    bool operator==(const FastFixed &o) const { return v == o.v; }
    bool operator!=(const FastFixed &o) const { return v != o.v; }
    bool operator<(const FastFixed &o) const { return v < o.v; }
    bool operator>(const FastFixed &o) const { return v > o.v; }
    bool operator<=(const FastFixed &o) const { return v <= o.v; }
    bool operator>=(const FastFixed &o) const { return v >= o.v; }

    friend FastFixed operator+(FastFixed a, FastFixed b) {
        return FastFixed::from_raw(a.v + b.v);
    }
    friend FastFixed operator-(FastFixed a, FastFixed b) {
        return FastFixed::from_raw(a.v - b.v);
    }
    friend FastFixed operator-(FastFixed x) {
        return FastFixed::from_raw(-x.v);
    }
    friend FastFixed operator*(FastFixed a, FastFixed b) {
        __int128 temp = static_cast<__int128>(a.v) * b.v;
        return FastFixed::from_raw(static_cast<underlying_type>(temp >> K));
    }
    friend FastFixed operator/(FastFixed a, FastFixed b) {
        __int128 temp = (static_cast<__int128>(a.v) << K) / b.v;
        return FastFixed::from_raw(static_cast<underlying_type>(temp));
    }

    FastFixed &operator+=(FastFixed b) { *this = *this + b; return *this; }
    FastFixed &operator-=(FastFixed b) { *this = *this - b; return *this; }
    FastFixed &operator*=(FastFixed b) { *this = *this * b; return *this; }
    FastFixed &operator/=(FastFixed b) { *this = *this / b; return *this; }

    friend ostream &operator<<(ostream &out, FastFixed x) {
        double val = static_cast<double>(x);
        return out << fixed << setprecision(2) << val;
    }
};

static constexpr array<pair<int, int>, 4> deltas{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

struct BaseSimulatorInterface {
    virtual void load_input(const string &filename) = 0;
    virtual void run_simulation(size_t T) = 0;
    virtual ~BaseSimulatorInterface() {}
};

template<typename PType, typename VType, typename VFlowType, size_t N, size_t M>
class FluidSimulator : public BaseSimulatorInterface {
public:
    FluidSimulator() {
        g = (PType)0.1;
        viscosity = (PType)0.0;
        for (size_t i = 0; i < 256; i++) {
            rho[i] = (PType)0;
        }
        for (size_t x = 0; x < N; x++) {
            for (size_t y = 0; y < M; y++) {
                field[x][y] = '#';
                p[x][y]      = (PType)0;
                old_p[x][y]  = (PType)0;
                dirs[x][y]   = 0;
            }
            field[x][M] = '\0';
        }
    }

    virtual ~FluidSimulator() = default;

    void load_input(const string &filename) override {
        ifstream in(filename);
        if(!in) {
            cerr << "Error: can't open file " << filename << "\n";
            exit(1);
        }

        size_t inputN, inputM;
        double gd, visc;
        if (!(in >> inputN >> inputM)) {
            cerr << "Error: Failed to read N and M.\n";
            exit(1);
        }
        if (inputN != N || inputM != M) {
            cerr << "Error: input sizes don't match the static sizes.\n";
            exit(1);
        }
        if (!(in >> gd >> visc)) {
            cerr << "Error: Failed to read gravity and viscosity.\n";
            exit(1);
        }

        int ccount;
        if (!(in >> ccount)) {
            cerr << "Error: Failed to read ccount.\n";
            exit(1);
        }
        for(int i = 0; i < ccount; i++){
            char c;
            double d;
            if(!(in >> c >> d)){
                cerr << "Error: Failed to read rho entry.\n";
                exit(1);
            }
            rho[(unsigned char)c] = (PType)d;
        }
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        for (size_t x = 0; x < N; x++) {
            for (size_t y = 0; y < M; y++) {
                char c;
                if(!in.get(c)){
                    cerr << "Error: Failed to read field.\n";
                    exit(1);
                }
                field[x][y] = c;
            }
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        g = (PType)gd;
        viscosity = (PType)visc;

        for (size_t x = 0; x < N; x++){
            for (size_t y = 0; y < M; y++){
                if(field[x][y] != '#'){
                    for(auto [dx, dy] : deltas){
                        int nx = (int)x + dx, ny = (int)y + dy;
                        if(nx >= 0 && nx < (int)N && ny >= 0 && ny < (int)M){
                            if(field[nx][ny] != '#') dirs[x][y]++;
                        }
                    }
                }
            }
        }
    }

    void run_simulation(size_t T) override {
        mt19937 rnd(1337);

        auto random01 = [&]() {
            double val = (double)(rnd() & 0xFFFF) / 65536.0;
            return (PType)val;
        };

        auto min_ = [&](auto a, auto b) {
            return (a < b) ? a : b;
        };

        auto parallel_for_x = [&](size_t startX, size_t endX, auto f) {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
            size_t total = endX - startX;
            size_t chunk = (total + num_threads - 1) / num_threads;

            vector<thread> threads;
            threads.reserve(num_threads);
            for (unsigned int t = 0; t < num_threads; t++) {
                size_t begin = startX + t*chunk;
                if (begin >= endX) break;
                size_t finish = std::min(begin + chunk, endX);

                threads.emplace_back([&, begin, finish, f](){
                    for (size_t x = begin; x < finish; x++) {
                        f(x);
                    }
                });
            }
            for (auto &th : threads) {
                if (th.joinable()) th.join();
            }
        };

        auto apply_viscosity = [&](auto &vf) {
            PType factor = (PType)(1.0 - (double)viscosity);
            parallel_for_x(0, N, [&](size_t x){
                for (size_t y = 0; y < M; y++) {
                    for (size_t i = 0; i < 4; i++) {
                        vf.v[x][y][i] *= factor;
                    }
                }
            });
        };

        static int last_use[N][M];
        mutex lastUseMutex;
        mutex velocityMutex;

        auto get_velocity_safe = [&](auto &vf, int x, int y, int dx, int dy) -> VFlowType {
            lock_guard<mutex> lk(velocityMutex);
            return vf.get(x, y, dx, dy);
        };
        auto add_velocity_safe = [&](auto &vf, int x, int y, int dx, int dy, VFlowType val){
            lock_guard<mutex> lk(velocityMutex);
            vf.add(x, y, dx, dy, val);
        };
        auto set_velocity_safe = [&](auto &vf, int x, int y, int dx, int dy, VFlowType val){
            lock_guard<mutex> lk(velocityMutex);
            auto &ref = vf.get(x, y, dx, dy);
            ref = val;
        };

        function< tuple<PType,bool,pair<int,int>>(int,int,PType,int(*)[M]) > propagate_flow;
        
        auto flow_impl = [&](auto &self, int x, int y, PType lim, int (*locLastUse)[M]) 
            -> tuple<PType,bool,pair<int,int>>
        {
            {
                lock_guard<mutex> lk(lastUseMutex);
                locLastUse[x][y] = UT - 1;
            }
            PType ret(0);
            for (auto [dx, dy] : deltas) {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= (int)N || ny < 0 || ny >= (int)M) continue;
                if (field[nx][ny] != '#') {
                    int lu;
                    {
                        lock_guard<mutex> lk(lastUseMutex);
                        lu = locLastUse[nx][ny];
                    }
                    if (lu < UT) {
                        auto cap  = get_velocity_safe(velocity, x, y, dx, dy);
                        auto flow = get_velocity_safe(velocity_flow, x, y, dx, dy);
                        if (flow == cap) continue;
                        auto vp = min_(lim, cap - flow);
                        {
                            lock_guard<mutex> lk(lastUseMutex);
                            if (locLastUse[nx][ny] == (UT - 1)) {
                                add_velocity_safe(velocity_flow, x, y, dx, dy, vp);
                                locLastUse[x][y] = UT;
                                return make_tuple(vp, true, pair<int,int>(nx, ny));
                            }
                        }
                        auto [t, prop, end] = self(self, nx, ny, vp, locLastUse);
                        ret += t;
                        if (prop) {
                            add_velocity_safe(velocity_flow, x, y, dx, dy, t);
                            {
                                lock_guard<mutex> lk(lastUseMutex);
                                locLastUse[x][y] = UT;
                            }
                            bool sameCell = (end == pair<int,int>(x, y));
                            return make_tuple(t, (prop && !sameCell), end);
                        }
                    }
                }
            }
            {
                lock_guard<mutex> lk(lastUseMutex);
                locLastUse[x][y] = UT;
            }
            return make_tuple(ret,false,pair<int,int>(0,0));
        };

        propagate_flow = [&](int x, int y, PType lim, int (*locLastUse)[M]) {
            return flow_impl(flow_impl, x, y, lim, locLastUse);
        };

        function<void(int,int,bool,int(*)[M])> propagate_stop;

        auto stop_impl = [&](auto &self, int x,int y,bool force,int (*locLastUse)[M]) -> void {
            if(!force) {
                bool stop = true;
                for(auto[dx,dy] : deltas){
                    int nx=x+dx, ny=y+dy;
                    if(nx<0||nx>=(int)N||ny<0||ny>=(int)M) continue;
                    if(field[nx][ny] != '#'){
                        int lu;
                        {
                            lock_guard<mutex> lk(lastUseMutex);
                            lu = locLastUse[nx][ny];
                        }
                        if(lu < UT - 1) {
                            auto v = get_velocity_safe(velocity, x,y, dx,dy);
                            if(v > (PType)0) {
                                stop=false;
                                break;
                            }
                        }
                    }
                }
                if(!stop) return;
            }
            {
                lock_guard<mutex> lk(lastUseMutex);
                locLastUse[x][y] = UT;
            }
            for(auto[dx,dy]:deltas){
                int nx=x+dx, ny=y+dy;
                if(nx<0||nx>=(int)N||ny<0||ny>=(int)M) continue;
                if(field[nx][ny]=='#') continue;
                {
                    lock_guard<mutex> lk(lastUseMutex);
                    if(locLastUse[nx][ny]==UT) continue;
                }
                auto v = get_velocity_safe(velocity, x,y, dx,dy);
                if(v>(PType)0) continue;
                self(self,nx,ny,false,locLastUse);
            }
        };

        propagate_stop = [&](int x,int y,bool force,int (*locLastUse)[M]) {
            return stop_impl(stop_impl, x,y,force,locLastUse);
        };

        function<bool(int,int,bool,int(*)[M])> propagate_move;

        struct ParticleParams {
            char type;
            PType cur_p;
            array<VFlowType,4> v;
            void swap_with(FluidSimulator &sim,int x,int y) {
                std::swap(sim.field[x][y], type);
                std::swap(sim.p[x][y], cur_p);
                for (size_t i=0;i<4;i++){
                    std::swap(sim.velocity.v[x][y][i], v[i]);
                }
            }
        };

        auto move_impl = [&](auto &self, int x,int y,bool is_first,int (*locLastUse)[M]) -> bool {
            {
                lock_guard<mutex> lk(lastUseMutex);
                locLastUse[x][y] = UT - (is_first?1:0);
            }
            bool ret=false;
            int nx=-1, ny=-1;

            do {
                array<PType,4> tres{};
                PType sum(0);
                for(size_t i=0;i<deltas.size();i++){
                    auto[dx,dy]=deltas[i];
                    int nx2=x+dx, ny2=y+dy;
                    if(nx2<0||nx2>=(int)N||ny2<0||ny2>=(int)M){
                        tres[i] = sum;
                        continue;
                    }
                    int lu;
                    {
                        lock_guard<mutex> lk(lastUseMutex);
                        lu = locLastUse[nx2][ny2];
                    }
                    if(field[nx2][ny2]=='#' || lu==UT){
                        tres[i]=sum;
                        continue;
                    }
                    auto v = get_velocity_safe(velocity, x,y, dx,dy);
                    if(v<(PType)0){
                        tres[i]=sum;
                        continue;
                    }
                    sum += v;
                    tres[i] = sum;
                }
                if(sum==(PType)0) break;

                PType pval = (PType)(random01()*sum);
                size_t d=0;
                for(; d<4; d++){
                    if(tres[d] > pval) break;
                }
                auto[dx,dy] = deltas[d];
                nx=x+dx; ny=y+dy;
                int luNext;
                {
                    lock_guard<mutex> lk(lastUseMutex);
                    luNext = locLastUse[nx][ny];
                }
                ret = (luNext == UT - 1) || self(self, nx, ny,false, locLastUse);
            } while(!ret);

            {
                lock_guard<mutex> lk(lastUseMutex);
                locLastUse[x][y] = UT;
            }
            for(auto[dx,dy]:deltas){
                int nx2=x+dx, ny2=y+dy;
                if(nx2<0||nx2>=(int)N||ny2<0||ny2>=(int)M) continue;
                if(field[nx2][ny2]=='#') continue;
                int lu2;
                {
                    lock_guard<mutex> lk(lastUseMutex);
                    lu2 = locLastUse[nx2][ny2];
                }
                if(lu2<UT-1){
                    auto v = get_velocity_safe(velocity, x,y, dx,dy);
                    if(v<(PType)0){
                        propagate_stop(nx2,ny2,false,locLastUse);
                    }
                }
            }
            if(ret && !is_first) {
                ParticleParams pp{};
                pp.type = field[x][y];
                pp.cur_p = p[x][y];
                for(size_t i=0; i<4; i++){
                    pp.v[i] = velocity.v[x][y][i];
                }
                pp.swap_with(*this,nx,ny);
                pp.swap_with(*this,x,y);
            }
            return ret;
        };

        propagate_move = [&](int x,int y,bool is_first,int (*locLastUse)[M]) {
            return move_impl(move_impl, x, y, is_first, locLastUse);
        };
   
        if(rho[(unsigned char)' ']==(PType)0) rho[(unsigned char)' ']=(PType)0.01;
        if(rho[(unsigned char)'.']==(PType)0) rho[(unsigned char)'.']=(PType)1000;

        for(size_t i=0; i<T; i++){
            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    if(field[x][y]=='#') continue;
                    if(x+1<N && field[x+1][y]!='#'){
                        add_velocity_safe(velocity, (int)x,(int)y, 1,0, g);
                    }
                }
            });

            parallel_for_x(0, N, [&](size_t x){
                memcpy(old_p[x], p[x], sizeof(p[x]));
            });

            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    if(field[x][y]=='#') continue;
                    for(auto [dx,dy] : deltas){
                        int nx = (int)x+dx, ny = (int)y+dy;
                        if(nx<0||nx>=(int)N||ny<0||ny>=(int)M) continue;
                        if(field[nx][ny]=='#') continue;

                        if(old_p[nx][ny] < old_p[x][y]) {
                            auto delta_p = old_p[x][y] - old_p[nx][ny];
                            auto force   = delta_p;
                            lock_guard<mutex> lk(velocityMutex);
                            auto &contr = velocity.get(nx, ny, -dx, -dy);
                            if(contr * rho[(unsigned char)field[nx][ny]] >= force) {
                                contr -= force / rho[(unsigned char)field[nx][ny]];
                            } else {
                                force -= contr*rho[(unsigned char)field[nx][ny]];
                                contr = (PType)0;
                                velocity.add((int)x,(int)y, dx,dy, force / rho[(unsigned char)field[x][y]]);
                                p[x][y] -= force / (PType)dirs[x][y];
                            }
                        }
                    }
                }
            });

            {
                lock_guard<mutex> lk(velocityMutex);
                memset(velocity_flow.v, 0, sizeof(velocity_flow.v));
            }
            parallel_for_x(0, N, [&](size_t xx){
                for(size_t yy=0; yy<M; yy++){
                    last_use[xx][yy] = -9999999;
                }
            });

            bool prop=false;
            do {
                UT += 2;
                prop = false;
                for(size_t x=0; x<N; x++){
                    for(size_t y=0; y<M; y++){
                        if(field[x][y]=='#') continue;
                        int lu;
                        {
                            lock_guard<mutex> lk(lastUseMutex);
                            lu = last_use[x][y];
                        }
                        if(lu == UT) continue;
                        auto [t, local_prop, _] = propagate_flow((int)x, (int)y, (PType)1, last_use);
                        if(t>0) prop=true;
                    }
                }
            } while(prop);

            apply_viscosity(velocity);
            apply_viscosity(velocity_flow);

            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    if(field[x][y]=='#') continue;
                    for(auto [dx,dy] : deltas){
                        auto old_v = get_velocity_safe(velocity, x,y, dx,dy);
                        auto new_v = get_velocity_safe(velocity_flow, x,y, dx,dy);
                        if(old_v > (PType)0){
                            auto diff = old_v - new_v;
                            set_velocity_safe(velocity, x,y, dx,dy, new_v);
                            auto force = diff * rho[(unsigned char)field[x][y]];
                            if(field[x][y]=='.') {
                                force *= (PType)0.8;
                            }
                            int nx = (int)x+dx, ny=(int)y+dy;
                            if(nx<0||nx>=(int)N||ny<0||ny>=(int)M||field[nx][ny]=='#'){
                                p[x][y] += force/(PType)dirs[x][y];
                            } else {
                                p[nx][ny]+= force/(PType)dirs[nx][ny];
                            }
                        }
                    }
                }
            });

            UT += 2;
            bool changed = false;
            for(size_t x=0; x<N; x++){
                for(size_t y=0; y<M; y++){
                    if(field[x][y]=='#') continue;
                    int lu;
                    {
                        lock_guard<mutex> lk(lastUseMutex);
                        lu = last_use[x][y];
                    }
                    if(lu == UT) continue;

                    auto move_prob = [&](int xx,int yy){
                        PType sum(0);
                        for (auto [ddx, ddy] : deltas) {
                            int nx2 = xx + ddx, ny2 = yy + ddy;
                            if(nx2<0||nx2>=(int)N||ny2<0||ny2>=(int)M) continue;
                            int lu2;
                            {
                                lock_guard<mutex> llk(lastUseMutex);
                                lu2 = last_use[nx2][ny2];
                            }
                            if(field[nx2][ny2]=='#' || lu2 == UT) continue;
                            auto v = get_velocity_safe(velocity, xx, yy, ddx, ddy);
                            if(v<(PType)0) continue;
                            sum += v;
                        }
                        return sum;
                    };

                    if(random01() < move_prob(x,y)){
                        if(propagate_move((int)x,(int)y,true,last_use)){
                            changed = true;
                        }
                    } else {
                        propagate_stop((int)x,(int)y,true,last_use);
                    }
                }
            }

            if(changed){
                cout << "Tick " << i << ":\n";
                for(size_t xx=0; xx<N; xx++){
                    cout << field[xx] << "\n";
                }
            }
        }
    }

private:
    char field[N][M+1];
    PType p[N][M]{}, old_p[N][M]{};
    PType rho[256]{};
    int dirs[N][M]{};
    int UT=0;
    PType g;
    PType viscosity;

    struct VectorField {
        array<VFlowType,4> v[N][M]{};
        VFlowType &add(int x,int y,int dx,int dy,VFlowType dv) {
            return get(x,y,dx,dy) += dv;
        }
        VFlowType &get(int x,int y,int dx,int dy) {
            for(size_t i=0; i<4; i++){
                if(deltas[i].first==dx && deltas[i].second==dy){
                    return v[x][y][i];
                }
            }
            return v[x][y][0];
        }
    } velocity, velocity_flow;
};

template<typename PType, typename VType, typename VFlowType>
class FluidSimulatorDynamic : public BaseSimulatorInterface {
public:
    FluidSimulatorDynamic()
        : N(0), M(0), UT(0), g((PType)0.1), viscosity((PType)0.0)
    {
        for (int i = 0; i < 256; i++) {
            rho[i] = (PType)0;
        }
    }

    FluidSimulatorDynamic(size_t n, size_t m)
        : N(n), M(m), UT(0), g((PType)0.1), viscosity((PType)0.0)
    {
        for (int i = 0; i < 256; i++) {
            rho[i] = (PType)0;
        }
        field.resize(N, std::vector<char>(M, '#'));
        p.resize(N, std::vector<PType>(M, (PType)0));
        old_p.resize(N, std::vector<PType>(M, (PType)0));
        dirs.resize(N, std::vector<int>(M, 0));

        velocity.init(N, M);
        velocity_flow.init(N, M);
    }

    virtual ~FluidSimulatorDynamic() = default;

    void load_input(const std::string &filename) override {
        std::ifstream in(filename);
        if(!in){
            std::cerr << "Error: can't open file " << filename << "\n";
            exit(1);
        }

        size_t inputN, inputM;
        if(!(in >> inputN >> inputM)){
            std::cerr << "Error: Failed to read N and M." << std::endl;
            exit(1);
        }
        N = inputN;
        M = inputM;

        field.resize(N, std::vector<char>(M, '#'));
        p.resize(N, std::vector<PType>(M, (PType)0));
        old_p.resize(N, std::vector<PType>(M, (PType)0));
        dirs.resize(N, std::vector<int>(M, 0));

        double gd, visc;
        if(!(in >> gd >> visc)){
            std::cerr << "Error: Failed to read gravity and viscosity." << std::endl;
            exit(1);
        }
        g = (PType)gd;
        viscosity = (PType)visc;

        int ccount;
        if(!(in >> ccount)){
            std::cerr << "Error: Failed to read ccount." << std::endl;
            exit(1);
        }
        for(int i = 0; i < ccount; i++){
            char c; 
            double d;
            if(!(in >> c >> d)){
                std::cerr << "Error: Failed to read rho entry " << i+1 << "." << std::endl;
                exit(1);
            }
            rho[(unsigned char)c] = (PType)d;
        }
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        for(size_t x = 0; x < N; x++){
            for(size_t y = 0; y < M; y++){
                char c;
                if(!in.get(c)){
                    std::cerr << "Error: Failed to read field[" << x << "][" << y << "]." << std::endl;
                    exit(1);
                }
                field[x][y] = c;
            }
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        for(size_t x = 0; x < N; x++){
            for(size_t y = 0; y < M; y++){
                if(field[x][y] == '#') {
                    dirs[x][y] = 0;
                } else {
                    int countDir = 0;
                    for(auto [dx, dy] : deltas){
                        int nx = (int)x + dx, ny = (int)y + dy;
                        if(nx >= 0 && (size_t)nx < N && ny >= 0 && (size_t)ny < M && field[nx][ny] != '#')
                            countDir++;
                    }
                    dirs[x][y] = countDir;
                }
            }
        }
        velocity.init(N, M);
        velocity_flow.init(N, M);
    }

    void run_simulation(size_t T) override {
        std::mt19937 rnd(1337);

        auto random01 = [&]() {
            double val = (double)(rnd() & 0xFFFF) / 65536.0;
            return (PType)val;
        };

        auto min_ = [&](auto a, auto b) {
            return (a < b) ? a : b;
        };

        auto parallel_for_x = [&](size_t startX, size_t endX, auto func){
            unsigned int num_threads = std::thread::hardware_concurrency();
            if(num_threads == 0) num_threads = 4;
            size_t total = endX - startX;
            size_t chunk = (total + num_threads - 1) / num_threads;

            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for(unsigned int t = 0; t < num_threads; t++){
                size_t begin = startX + t*chunk;
                if(begin >= endX) break;
                size_t finish = std::min(begin + chunk, endX);

                threads.emplace_back([&, begin, finish, func](){
                    for(size_t x = begin; x < finish; x++){
                        func(x);
                    }
                });
            }
            for(auto &th : threads){
                if(th.joinable()) th.join();
            }
        };

        auto apply_viscosity = [&](auto &vf){
            PType factor = (PType)(1.0 - (double)viscosity);
            parallel_for_x(0, N, [&](size_t x){
                for(size_t y = 0; y < M; y++){
                    for(size_t i = 0; i < 4; i++){
                        vf.v[x][y][i] *= factor;
                    }
                }
            });
        };

        std::vector<std::vector<int>> last_use(N, std::vector<int>(M, -9999999));
        std::mutex lastUseMutex;
        std::mutex velocityMutex;

        auto get_velocity_safe = [&](auto &vf, int x, int y, int dx, int dy) -> VFlowType {
            std::lock_guard<std::mutex> lk(velocityMutex);
            return vf.get(x, y, dx, dy);
        };
        auto add_velocity_safe = [&](auto &vf, int x, int y, int dx, int dy, VFlowType val){
            std::lock_guard<std::mutex> lk(velocityMutex);
            vf.add(x, y, dx, dy, val);
        };
        auto set_velocity_safe = [&](auto &vf, int x, int y, int dx, int dy, VFlowType val){
            std::lock_guard<std::mutex> lk(velocityMutex);
            vf.set(x, y, dx, dy, val);
        };

        std::function<std::tuple<PType,bool,std::pair<int,int>>(int,int,PType)> propagate_flow;
        auto flow_impl = [&](auto &self, int x, int y, PType lim) 
            -> std::tuple<PType,bool,std::pair<int,int>>
        {
            {
                std::lock_guard<std::mutex> lk(lastUseMutex);
                last_use[x][y] = UT - 1;
            }
            PType ret(0);
            for(auto [dx, dy] : deltas){
                int nx = x + dx, ny = y + dy;
                if(nx < 0 || nx >= (int)N || ny < 0 || ny >= (int)M) continue;
                if(field[nx][ny] == '#') continue;

                int lu;
                {
                    std::lock_guard<std::mutex> lk(lastUseMutex);
                    lu = last_use[nx][ny];
                }
                if(lu < UT){
                    auto cap  = get_velocity_safe(velocity, x,y, dx,dy);
                    auto flow = get_velocity_safe(velocity_flow, x,y, dx,dy);
                    if(flow == cap) continue;
                    auto vp = min_(lim, cap - flow);
                    {
                        std::lock_guard<std::mutex> lk(lastUseMutex);
                        if(last_use[nx][ny] == (UT - 1)){
                            add_velocity_safe(velocity_flow, x,y, dx,dy, vp);
                            last_use[x][y] = UT;
                            return std::make_tuple(vp, true, std::pair<int,int>(nx,ny));
                        }
                    }
                    auto [t, prop, end] = self(self, nx, ny, vp);
                    ret += t;
                    if(prop){
                        add_velocity_safe(velocity_flow, x,y, dx,dy, t);
                        {
                            std::lock_guard<std::mutex> lk(lastUseMutex);
                            last_use[x][y] = UT;
                        }
                        bool sameCell = (end == std::pair<int,int>(x,y));
                        return std::make_tuple(t, (prop && !sameCell), end);
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lk(lastUseMutex);
                last_use[x][y] = UT;
            }
            return std::make_tuple(ret, false, std::pair<int,int>(0,0));
        };

        propagate_flow = [&](int x, int y, PType lim){
            return flow_impl(flow_impl, x, y, lim);
        };

        std::function<void(int,int,bool)> propagate_stop;
        auto stop_impl = [&](auto &self, int x,int y,bool force){
            if(!force){
                bool stop = true;
                for(auto[dx,dy]: deltas){
                    int nx = x+dx, ny = y+dy;
                    if(nx<0||nx>=(int)N||ny<0||ny>=(int)M) continue;
                    if(field[nx][ny] != '#'){
                        int lu;
                        {
                            std::lock_guard<std::mutex> lk(lastUseMutex);
                            lu = last_use[nx][ny];
                        }
                        if(lu < UT - 1){
                            auto v = get_velocity_safe(velocity, x,y, dx,dy);
                            if(v>(PType)0){
                                stop = false;
                                break;
                            }
                        }
                    }
                }
                if(!stop) return;
            }
            {
                std::lock_guard<std::mutex> lk(lastUseMutex);
                last_use[x][y] = UT;
            }
            for(auto[dx,dy]:deltas){
                int nx=x+dx, ny=y+dy;
                if(nx<0||nx>=(int)N||ny<0||ny>=(int)M) continue;
                if(field[nx][ny]=='#') continue;
                {
                    std::lock_guard<std::mutex> lk(lastUseMutex);
                    if(last_use[nx][ny] == UT) continue;
                }
                auto v = get_velocity_safe(velocity, x,y, dx,dy);
                if(v>(PType)0) continue;
                self(self,nx,ny,false);
            }
        };
        propagate_stop = [&](int x,int y,bool force){
            return stop_impl(stop_impl, x,y,force);
        };

        struct ParticleParams {
            char type;
            PType cur_p;
            std::array<VFlowType,4> v;
            void swap_with(FluidSimulatorDynamic &sim, int x, int y) {
                std::swap(sim.field[x][y], type);
                std::swap(sim.p[x][y], cur_p);
                for(size_t i=0; i<4; i++){
                    std::swap(sim.velocity.v[x][y][i], v[i]);
                }
            }
        };

        std::function<bool(int,int,bool)> propagate_move;
        auto move_impl = [&](auto &self,int x,int y,bool is_first)->bool {
            {
                std::lock_guard<std::mutex> lk(lastUseMutex);
                last_use[x][y] = UT - (is_first?1:0);
            }
            bool ret = false;
            int nx=-1, ny=-1;

            do {
                std::array<PType,4> tres{};
                PType sum(0);
                for(size_t i=0; i<deltas.size(); i++){
                    auto[dx,dy] = deltas[i];
                    int nx2 = x+dx, ny2 = y+dy;
                    if(nx2<0||nx2>=(int)N||ny2<0||ny2>=(int)M){
                        tres[i] = sum;
                        continue;
                    }
                    int lu;
                    {
                        std::lock_guard<std::mutex> lk(lastUseMutex);
                        lu = last_use[nx2][ny2];
                    }
                    if(field[nx2][ny2] == '#' || lu==UT){
                        tres[i] = sum;
                        continue;
                    }
                    auto v = get_velocity_safe(velocity, x,y, dx,dy);
                    if(v<(PType)0){
                        tres[i]=sum;
                        continue;
                    }
                    sum += v;
                    tres[i] = sum;
                }
                if(sum==(PType)0) break;

                PType pval = (PType)(random01()*sum);
                size_t d = 0;
                for(; d<4; d++){
                    if(tres[d] > pval) break;
                }
                auto [mdx, mdy] = deltas[d];
                nx = x + mdx; ny = y + mdy;
                int luNext;
                {
                    std::lock_guard<std::mutex> lk(lastUseMutex);
                    luNext = last_use[nx][ny];
                }
                ret = (luNext == UT-1) || self(self, nx, ny, false);
            } while(!ret);

            {
                std::lock_guard<std::mutex> lk(lastUseMutex);
                last_use[x][y] = UT;
            }
            for(auto[dx,dy] : deltas){
                int nx2 = x+dx, ny2 = y+dy;
                if(nx2<0||nx2>=(int)N||ny2<0||ny2>=(int)M) continue;
                if(field[nx2][ny2] == '#') continue;
                int lu2;
                {
                    std::lock_guard<std::mutex> lk(lastUseMutex);
                    lu2 = last_use[nx2][ny2];
                }
                if(lu2 < UT-1){
                    auto v = get_velocity_safe(velocity, x,y, dx,dy);
                    if(v<(PType)0){
                        propagate_stop(nx2,ny2,false);
                    }
                }
            }

            if(ret && !is_first){
                ParticleParams pp{};
                pp.type = field[x][y];
                pp.cur_p = p[x][y];
                for(size_t i=0; i<4; i++){
                    pp.v[i] = velocity.v[x][y][i];
                }
                pp.swap_with(*this, nx, ny);
                pp.swap_with(*this, x, y);
            }
            return ret;
        };
        propagate_move = [&](int x,int y,bool is_first){
            return move_impl(move_impl, x,y,is_first);
        };

        if(rho[(unsigned char)' '] == (PType)0) {
            rho[(unsigned char)' '] = (PType)0.01;
        }
        if(rho[(unsigned char)'.'] == (PType)0){
            rho[(unsigned char)'.'] = (PType)1000;
        }

        for(size_t i=0; i<T; i++){
            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    if(field[x][y] == '#') continue;
                    if(x+1 < N && field[x+1][y] != '#'){
                        add_velocity_safe(velocity, (int)x,(int)y, 1,0, g);
                    }
                }
            });

            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    old_p[x][y] = p[x][y];
                }
            });

            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    if(field[x][y] == '#') continue;
                    for(auto [dx,dy] : deltas){
                        int nx = (int)x + dx, ny = (int)y + dy;
                        if(nx<0||nx>=(int)N||ny<0||ny>=(int)M) continue;
                        if(field[nx][ny] == '#') continue;

                        if(old_p[nx][ny] < old_p[x][y]){
                            auto delta_p = old_p[x][y] - old_p[nx][ny];
                            auto force = delta_p;

                            std::lock_guard<std::mutex> lk(velocityMutex);
                            auto &contr = velocity.v[nx][ny][dir_index(-dx,-dy)];
                            if(contr * rho[(unsigned char)field[nx][ny]] >= force){
                                contr -= force / rho[(unsigned char)field[nx][ny]];
                            } else {
                                force -= contr * rho[(unsigned char)field[nx][ny]];
                                contr = (PType)0;
                                velocity.add((int)x,(int)y, dx,dy, force / rho[(unsigned char)field[x][y]]);
                                p[x][y] -= force / (PType)dirs[x][y];
                            }
                        }
                    }
                }
            });

            {
                std::lock_guard<std::mutex> lk(velocityMutex);
                for(size_t x=0; x<N; x++){
                    for(size_t y=0; y<M; y++){
                        for(int i=0; i<4; i++){
                            velocity_flow.v[x][y][i] = (VFlowType)0;
                        }
                    }
                }
            }
            parallel_for_x(0, N, [&](size_t xx){
                for(size_t yy=0; yy<M; yy++){
                    last_use[xx][yy] = -9999999;
                }
            });

            bool prop = false;
            do {
                UT += 2;
                prop = false;
                for(size_t x=0; x<N; x++){
                    for(size_t y=0; y<M; y++){
                        if(field[x][y] == '#') continue;
                        int lu;
                        {
                            std::lock_guard<std::mutex> lk(lastUseMutex);
                            lu = last_use[x][y];
                        }
                        if(lu == UT) continue;

                        auto [t, local_prop, _] = propagate_flow((int)x, (int)y, (PType)1);
                        if(t > 0) prop=true;
                    }
                }
            } while(prop);

            apply_viscosity(velocity);
            apply_viscosity(velocity_flow);

            parallel_for_x(0, N, [&](size_t x){
                for(size_t y=0; y<M; y++){
                    if(field[x][y] == '#') continue;
                    for(auto [dx,dy] : deltas){
                        auto old_v = get_velocity_safe(velocity, (int)x,(int)y, dx,dy);
                        auto new_v = get_velocity_safe(velocity_flow, (int)x,(int)y, dx,dy);
                        if(old_v > (PType)0){
                            auto diff = old_v - new_v;
                            set_velocity_safe(velocity, (int)x,(int)y, dx,dy, new_v);

                            auto force = diff * rho[(unsigned char)field[x][y]];
                            if(field[x][y] == '.'){
                                force *= (PType)0.8;
                            }
                            int nx = (int)x+dx, ny = (int)y+dy;
                            if(nx<0||nx>=(int)N||ny<0||ny>=(int)M||field[nx][ny]=='#'){
                                p[x][y] += force / (PType)dirs[x][y];
                            } else {
                                p[nx][ny] += force / (PType)dirs[nx][ny];
                            }
                        }
                    }
                }
            });

            UT += 2;
            bool changed = false;
            for(size_t x=0; x<N; x++){
                for(size_t y=0; y<M; y++){
                    if(field[x][y] == '#') continue;
                    int lu;
                    {
                        std::lock_guard<std::mutex> lk(lastUseMutex);
                        lu = last_use[x][y];
                    }
                    if(lu == UT) continue;

                    auto move_prob = [&](int xx, int yy){
                        PType sum(0);
                        for(auto[ddx, ddy] : deltas){
                            int nx2 = xx + ddx, ny2 = yy + ddy;
                            if(nx2<0||nx2>=(int)N||ny2<0||ny2>=(int)M) continue;
                            int lu2;
                            {
                                std::lock_guard<std::mutex> llk(lastUseMutex);
                                lu2 = last_use[nx2][ny2];
                            }
                            if(field[nx2][ny2] == '#' || lu2 == UT) continue;
                            auto v = get_velocity_safe(velocity, xx,yy, ddx,ddy);
                            if(v < (PType)0) continue;
                            sum += v;
                        }
                        return sum;
                    };

                    if(random01() < move_prob((int)x,(int)y)){
                        if(propagate_move((int)x,(int)y,true)){
                            changed = true;
                        }
                    } else {
                        propagate_stop((int)x,(int)y,true);
                    }
                }
            }

            if(changed){
                std::cout << "Tick " << i << ":\n";
                for(size_t xx=0; xx<N; xx++){
                    for(size_t yy=0; yy<M; yy++){
                        std::cout << field[xx][yy];
                    }
                    std::cout << "\n";
                }
            }
        }
    }

private:
    int dir_index(int dx, int dy) {
        for(int i=0; i<4; i++){
            if(deltas[i].first == dx && deltas[i].second == dy){
                return i;
            }
        }
        return 0;
    }

    size_t N, M;
    std::vector<std::vector<char>>   field;
    std::vector<std::vector<PType>>  p, old_p;
    std::array<PType, 256>           rho{};
    std::vector<std::vector<int>>    dirs;
    int UT;
    PType g, viscosity;

    static constexpr std::array<std::pair<int,int>,4> deltas{{{-1,0},{1,0},{0,-1},{0,1}}};

    struct VectorField {
        std::vector<std::vector< std::array<VFlowType,4> >> v;

        VectorField() = default;

        void init(size_t n, size_t m){
            v.resize(n);
            for(auto &row : v){
                row.resize(m);
            }
        }

        VFlowType &add(int x,int y,int dx,int dy, VFlowType dv){
            return get(x,y,dx,dy) += dv;
        }
        VFlowType &get(int x,int y,int dx,int dy){
            for(int i=0; i<4; i++){
                if(deltas[i].first==dx && deltas[i].second==dy){
                    return v[x][y][i];
                }
            }
            return v[x][y][0];
        }
        void set(int x,int y,int dx,int dy, VFlowType val){
            for(int i=0; i<4; i++){
                if(deltas[i].first==dx && deltas[i].second==dy){
                    v[x][y][i] = val;
                    return;
                }
            }
            v[x][y][0] = val;
        }
    } velocity, velocity_flow;
};

struct SimulatorImplBase : BaseSimulatorInterface {
    virtual ~SimulatorImplBase(){}
};

template<typename PType, typename VType, typename VFlowType, size_t N, size_t M>
struct SimulatorImpl : SimulatorImplBase {
    FluidSimulator<PType, VType, VFlowType, N, M> sim;
    void load_input(const string &filename) override {
        sim.load_input(filename);
    }
    void run_simulation(size_t T) override {
        sim.run_simulation(T);
    }
};

template<typename PType, typename VType, typename VFlowType>
struct SimulatorImplDynamic : SimulatorImplBase {
    unique_ptr<FluidSimulatorDynamic<PType,VType,VFlowType>> sim;
    SimulatorImplDynamic(size_t n,size_t m) {
        sim = make_unique<FluidSimulatorDynamic<PType,VType,VFlowType>>(n, m);
    }
    void load_input(const string &filename) override {
        sim->load_input(filename);
    }
    void run_simulation(size_t T) override {
        sim->run_simulation(T);
    }
};

static std::unique_ptr<BaseSimulatorInterface> create_simulator(
    const std::string &p_type_str,
    const std::string &v_type_str,
    const std::string &v_flow_type_str,
    size_t inputN, 
    size_t inputM)
{
    if ((inputN == 36 && inputM == 84)) {
        if (p_type_str == "FLOAT" && v_type_str == "FLOAT" && v_flow_type_str == "FLOAT") {
            return std::make_unique<SimulatorImpl<FLOAT_t, FLOAT_t, FLOAT_t, 36, 84>>();
        }
        if (p_type_str == "DOUBLE" && v_type_str == "DOUBLE" && v_flow_type_str == "DOUBLE") {
            return std::make_unique<SimulatorImpl<DOUBLE_t, DOUBLE_t, DOUBLE_t, 36, 84>>();
        }
        if (p_type_str == "FIXED(32,16)" && 
            v_type_str == "FIXED(32,16)" && 
            v_flow_type_str == "FIXED(32,16)")
        {
            return std::make_unique<SimulatorImpl<Fixed<32,16>, Fixed<32,16>, Fixed<32,16>, 36, 84>>();
        }
        if (p_type_str == "FAST_FIXED(32,16)" && 
            v_type_str == "FAST_FIXED(32,16)" && 
            v_flow_type_str == "FAST_FIXED(32,16)")
        {
            return std::make_unique<SimulatorImpl<FastFixed<32,16>, FastFixed<32,16>, FastFixed<32,16>, 36, 84>>();
        }

        throw std::runtime_error("No matching types for S(36,84)");
    }
    else if ((inputN == 14 && inputM == 5)) {
        if (p_type_str == "FLOAT" && v_type_str == "FLOAT" && v_flow_type_str == "FLOAT") {
            return std::make_unique<SimulatorImpl<FLOAT_t, FLOAT_t, FLOAT_t, 14, 5>>();
        }
        if (p_type_str == "DOUBLE" && v_type_str == "DOUBLE" && v_flow_type_str == "DOUBLE") {
            return std::make_unique<SimulatorImpl<DOUBLE_t, DOUBLE_t, DOUBLE_t, 14, 5>>();
        }
        if (p_type_str == "FIXED(32,16)" && 
            v_type_str == "FIXED(32,16)" && 
            v_flow_type_str == "FIXED(32,16)")
        {
            return std::make_unique<SimulatorImpl<Fixed<32,16>, Fixed<32,16>, Fixed<32,16>, 14, 5>>();
        }
        if (p_type_str == "FAST_FIXED(32,16)" && 
            v_type_str == "FAST_FIXED(32,16)" && 
            v_flow_type_str == "FAST_FIXED(32,16)")
        {
            return std::make_unique<SimulatorImpl<FastFixed<32,16>, FastFixed<32,16>, FastFixed<32,16>, 14, 5>>();
        }

        throw std::runtime_error("No matching types for S(14,5)");
    }
    else {
        if (p_type_str == "FLOAT" && v_type_str == "FLOAT" && v_flow_type_str == "FLOAT") {
            return std::make_unique<SimulatorImplDynamic<FLOAT_t, FLOAT_t, FLOAT_t>>(inputN, inputM);
        }
        if (p_type_str == "DOUBLE" && v_type_str == "DOUBLE" && v_flow_type_str == "DOUBLE") {
            return std::make_unique<SimulatorImplDynamic<DOUBLE_t, DOUBLE_t, DOUBLE_t>>(inputN, inputM);
        }
        if (p_type_str == "FIXED(32,16)" && 
            v_type_str == "FIXED(32,16)" && 
            v_flow_type_str == "FIXED(32,16)")
        {
            return std::make_unique<SimulatorImplDynamic<Fixed<32,16>, Fixed<32,16>, Fixed<32,16>>>(inputN, inputM);
        }
        if (p_type_str == "FAST_FIXED(32,16)" && 
            v_type_str == "FAST_FIXED(32,16)" && 
            v_flow_type_str == "FAST_FIXED(32,16)")
        {
            return std::make_unique<SimulatorImplDynamic<FastFixed<32,16>, FastFixed<32,16>, FastFixed<32,16>>>(inputN, inputM);
        }

        throw std::runtime_error("No matching types for dynamic size");
    }
}

int main(int argc, char** argv) {
    // ios::sync_with_stdio(false);
    cin.tie(nullptr);

    std::string filename = "input_.txt";

    std::string p_type_str = "FLOAT";
    std::string v_type_str = "FLOAT";
    std::string v_flow_type_str = "FLOAT";

    size_t inputN, inputM;
    {
        std::ifstream fin(filename);
        if (!fin) {
            cerr << "Cannot open " << filename << endl;
            return 1;
        }
        if (!(fin >> inputN >> inputM)) {
            cerr << "Failed to read N,M\n";
            return 1;
        }
    }
    
    unique_ptr<BaseSimulatorInterface> sim;
    try {
        sim = create_simulator(p_type_str, v_type_str, v_flow_type_str, inputN, inputM);
    } catch(const std::runtime_error &e){
        cerr << e.what() << endl;
        return 1;
    }

    sim->load_input(filename);
    sim->run_simulation(1700);

    return 0;
}
