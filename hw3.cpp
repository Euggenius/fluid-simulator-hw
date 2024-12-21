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
    static_assert(N > K, "N must be greater than K");
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
        g = static_cast<PType>(0.1);
        viscosity = static_cast<PType>(0.0);
        for (auto &r : rho) r = static_cast<PType>(0);
        for (size_t x = 0; x < N; x++) {
            for (size_t y = 0; y < M; y++) {
                field[x][y] = '#';
                p[x][y] = static_cast<PType>(0);
                old_p[x][y] = static_cast<PType>(0);
            }
        }
    }

    void load_input(const string &filename) {
        ifstream in(filename);
        if(!in){
            cerr<<"Error: can't open file "<<filename<<"\n";
            exit(1);
        }

        size_t inputN, inputM;
        double gd, visc;
        in >> inputN >> inputM;
        in >> gd;
        in >> visc;
        int ccount;
        in >> ccount;
        for(int i=0;i<ccount;i++){
            char c;double d; in>>c>>d;
            rho[(unsigned char)c]=(PType)d;
        }

        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        for(size_t x=0;x<N;x++){
            for(size_t y=0;y<M;y++){
                char c; in.get(c);
                field[x][y]=c;
            }
            field[x][M]='\0';
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        g=(PType)gd;
        viscosity=(PType)visc;

        for(size_t x=0;x<N;x++){
            for(size_t y=0;y<M;y++){
                dirs[x][y]=0;
                if(field[x][y]=='#') continue;
                for (auto [dx,dy]:deltas){
                    int nx=(int)x+dx, ny=(int)y+dy;
                    if(nx>=0&&(size_t)nx<N && ny>=0 && (size_t)ny<M && field[nx][ny]!='#')
                        dirs[x][y]++;
                }
            }
        }
    }

    void run_simulation(size_t T) override {
        unsigned int num_threads = thread::hardware_concurrency();
        if(num_threads == 0) num_threads = 4;
        cout << "Number of threads: " << num_threads << endl;

        for(size_t tick = 0; tick < T; tick++) {
            cout << "Tick: " << tick << endl;

            {
                size_t chunk_size = N / num_threads;
                size_t remainder = N % num_threads;
                vector<thread> threads;
                size_t current_x = 0;

                auto apply_gravity = [&](size_t start_x, size_t end_x) {
                    for(size_t x = start_x; x < end_x; x++) {
                        for(size_t y = 0; y < M; y++) {
                            if(field[x][y] == '#') continue;
                            if(x + 1 < N && field[x + 1][y] != '#') {
                                lock_guard<mutex> lock(velocity_mutex[x][y]);
                                velocity.v[x][y][1] += g;
                            }
                        }
                    }
                };

                for(unsigned int t = 0; t < num_threads; t++) {
                    size_t start_x = current_x;
                    size_t end_x = start_x + chunk_size + (t < remainder ? 1 : 0);
                    threads.emplace_back(thread(apply_gravity, start_x, end_x));
                    current_x = end_x;
                }

                for(auto &th : threads){
                    if(th.joinable()) th.join();
                }
            }

            {
                size_t chunk_size = N / num_threads;
                size_t remainder = N % num_threads;
                vector<thread> threads;
                size_t current_x = 0;

                auto update_pressure = [&](size_t start_x, size_t end_x) {
                    for(size_t x = start_x; x < end_x; x++) {
                        for(size_t y = 0; y < M; y++) {
                            if(field[x][y] == '#') continue;
                            for(auto [dx, dy] : deltas) {
                                int nx = static_cast<int>(x) + dx, ny = static_cast<int>(y) + dy;
                                if(nx < 0 || static_cast<size_t>(nx) >= N || ny < 0 || static_cast<size_t>(ny) >= M) continue;
                                if(field[nx][ny] != '#' && old_p[nx][ny] < p[x][y]) {
                                    auto delta_p = p[x][y] - old_p[nx][ny];
                                    auto force = delta_p;

                                    PType neighbors_fixed = static_cast<PType>(dirs[x][y]);

                                    if(neighbors_fixed > 0) {
                                        PType force_div = force / neighbors_fixed;
                                        lock_guard<mutex> lock_p(p_mutex[x][y]);
                                        p[x][y] -= force_div;
                                    }
                                }
                            }
                        }
                    }
                };

                for(unsigned int t = 0; t < num_threads; t++) {
                    size_t start_x = current_x;
                    size_t end_x = start_x + chunk_size + (t < remainder ? 1 : 0);
                    threads.emplace_back(thread(update_pressure, start_x, end_x));
                    current_x = end_x;
                }

                for(auto &th : threads){
                    if(th.joinable()) th.join();
                }
            }

            {
                size_t chunk_size = N / num_threads;
                size_t remainder = N % num_threads;
                vector<thread> threads;
                size_t current_x = 0;

                auto spread_fluid = [&](size_t start_x, size_t end_x) {
                    for(size_t x = start_x; x < end_x; x++) {
                        for(size_t y = 0; y < M; y++) {
                            if(field[x][y] == '#' || field[x][y] == ' ') continue;
                            if(x + 1 < N && field[x + 1][y] == ' ') {
                                field[x + 1][y] = field[x][y];
                                field[x][y] = ' ';
                            }
                        }
                    }
                };

                for(unsigned int t = 0; t < num_threads; t++) {
                    size_t start_x = current_x;
                    size_t end_x = start_x + chunk_size + (t < remainder ? 1 : 0);
                    threads.emplace_back(thread(spread_fluid, start_x, end_x));
                    current_x = end_x;
                }

                for(auto &th : threads){
                    if(th.joinable()) th.join();
                }
            }
            for(size_t x = 0; x < N; x++) {
                for(size_t y = 0; y < M; y++) {
                    cout << field[x][y];
                }
                cout << "\n";
            }

            for(size_t x = 0; x < N; x++) {
                for(size_t y = 0; y < M; y++) {
                    old_p[x][y] = p[x][y];
                }
            }
        }
    }

private:
    char field[N][M];
    array<array<PType, M>, N> p, old_p;
    array<PType, 256> rho;
    array<array<int, M>, N> dirs;
    int UT = 0;
    PType g, viscosity;
    struct VectorField { array<VFlowType, 4> v[N][M]; } velocity, velocity_flow;
    mutex velocity_mutex[N][M], p_mutex[N][M];
};

template<typename PType, typename VType, typename VFlowType>
unique_ptr<BaseSimulatorInterface> create_simulator(
    const string &p_type_str,
    const string &v_type_str,
    const string &v_flow_type_str,
    size_t inputN, size_t inputM)
{
    if((inputN == 36 && inputM == 84)) {
        if(p_type_str == "FLOAT" && v_type_str == "FLOAT" && v_flow_type_str == "FLOAT") {
            return make_unique<FluidSimulator<PType, VType, VFlowType, 36, 84>>();
        }
        if(p_type_str == "DOUBLE" && v_type_str == "DOUBLE" && v_flow_type_str == "DOUBLE") {
            return make_unique<FluidSimulator<PType, VType, VFlowType, 36, 84>>();
        }
        if(p_type_str == "FIXED(32,16)" && v_type_str == "FIXED(32,16)" && v_flow_type_str == "FIXED(32,16)") {
            return make_unique<FluidSimulator<Fixed<32,16>, Fixed<32,16>, Fixed<32,16>, 36, 84>>();
        }
        if(p_type_str == "FAST_FIXED(32,16)" && v_type_str == "FAST_FIXED(32,16)" && v_flow_type_str == "FAST_FIXED(32,16)") {
            return make_unique<FluidSimulator<FastFixed<32,16>, FastFixed<32,16>, FastFixed<32,16>, 36, 84>>();
        }

        throw runtime_error("No matching types for S(36,84)");
    }
    else if((inputN == 14 && inputM == 5)){
        if(p_type_str == "FLOAT" && v_type_str == "FLOAT" && v_flow_type_str == "FLOAT") {
            return make_unique<FluidSimulator<PType, VType, VFlowType, 14, 5>>();
        }
        if(p_type_str == "DOUBLE" && v_type_str == "DOUBLE" && v_flow_type_str == "DOUBLE") {
            return make_unique<FluidSimulator<PType, VType, VFlowType, 14, 5>>();
        }
        if(p_type_str == "FIXED(32,16)" && v_type_str == "FIXED(32,16)" && v_flow_type_str == "FIXED(32,16)") {
            return make_unique<FluidSimulator<Fixed<32,16>, Fixed<32,16>, Fixed<32,16>, 14, 5>>();
        }
        if(p_type_str == "FAST_FIXED(32,16)" && v_type_str == "FAST_FIXED(32,16)" && v_flow_type_str == "FAST_FIXED(32,16)") {
            return make_unique<FluidSimulator<FastFixed<32,16>, FastFixed<32,16>, FastFixed<32,16>, 14, 5>>();
        }

        throw runtime_error("No matching types for S(14,5)");
    }
    else {
        throw runtime_error("No matching types for dynamic size");
    }
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string p_type_str = "FLOAT", v_type_str = "FLOAT", v_flow_type_str = "FLOAT", filename = "input.txt";
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg.rfind("--p-type=", 0) == 0) p_type_str = arg.substr(9);
        else if (arg.rfind("--v-type=", 0) == 0) v_type_str = arg.substr(9);
        else if (arg.rfind("--v-flow-type=", 0) == 0) v_flow_type_str = arg.substr(14);
        else filename = arg;
    }
    ifstream in(filename);
    if (!in){
        cerr << "Error: cannot open " << filename << "\n";
        return 1;
    }
    size_t inputN, inputM;
    in>>inputN>>inputM;
    in.close();

    unique_ptr<BaseSimulatorInterface> sim=create_simulator<FLOAT_t, FLOAT_t, FLOAT_t>(p_type_str, v_type_str, v_flow_type_str, inputN, inputM);
    sim->load_input(filename);
    sim->run_simulation(1500);

    return 0;
}
