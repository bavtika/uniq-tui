/*
 * vanity.cu — Ed25519 vanity miner (GPU): incremental +8·G or seed-derived key path.
 *
 * Interactive:
 *   ./cuda_ed25519_vanity [--watch [path]] [--user-point <64-hex>] [--phantom] <pattern>...
 *   MATCH / MATCH_SK:  <pi>: <b58> - <64-hex scalar or seed>
 *
 * Worker mode (address = base58(pub); secret = base58(32-byte seed || 32-byte pub)):
 *   --prefix PAT | --suffix PAT
 *   --case-sensitive true|false
 *   --stop-after-keys N  --max-iterations M  --attempts-per-execution K  --quiet
 * Multi:
 *   --patterns id1=pat1,id2=pat2  --is-prefix true|false  (+ same worker flags)
 */

#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <climits>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <sys/stat.h>

#include "curand_kernel.h"
#include "ed25519.h"

#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"

#define BATCH_W     64
#define ATTEMPTS    (50000 * BATCH_W)

#define MAX_PATTERNS    16
#define MAX_PATTERN_LEN 44

__constant__ char g_patterns[MAX_PATTERNS][MAX_PATTERN_LEN + 1];
__constant__ int  g_pattern_lens[MAX_PATTERNS];
__constant__ int  g_pattern_count;

__constant__ unsigned char g_user_point[32];
__constant__ int           g_split_key_mode;

__constant__ uint8_t g_range_lo[MAX_PATTERNS][32];
__constant__ uint8_t g_range_hi[MAX_PATTERNS][32];
__constant__ int     g_range_valid[MAX_PATTERNS];

__constant__ ge_cached g_step_cached;

__constant__ int g_phantom_mode;
__constant__ int g_suffix_mode;
__constant__ int g_case_sensitive;
__constant__ int g_worker_output;
__constant__ int g_batches; /* 0 → ATTEMPTS / BATCH_W */

__device__ int g_match_flag;
__device__ int g_match_pi;
__device__ unsigned char g_match_seed[32];
__device__ unsigned char g_match_pub[32];

typedef struct { curandState *states[8]; } gpu_config;

struct VanityOpts {
    bool quiet = false;
    bool worker_output = false;
    bool worker_multi = false;
    bool phantom = false;
    bool suffix = false;
    bool no_range_prefilter = false;
    int stop_after_keys = 0;
    long long max_iterations = LLONG_MAX;
    uint64_t attempts_per_execution = 0;
};

void            vanity_setup(gpu_config &v, int total, int block);
void            vanity_run  (gpu_config &v, int total, int block,
                             bool watch, const char *watch_path,
                             time_t watch_mtime0, const std::string &patterns_sig0,
                             const VanityOpts &opts,
                             const std::vector<std::string> &patterns_live,
                             const std::vector<std::string> &request_ids);
void __global__ vanity_init (curandState *state);
void __global__ vanity_scan (curandState *state);
bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz);

/* ── Host base58: same algorithm as __device__ b58enc (64-byte keys need large buf) ─ */
static bool host_b58_encode(const uint8_t *bin, size_t binsz, char *b58, size_t b58cap) {
    static const char b58d[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    size_t zcount = 0;
    while (zcount < binsz && bin[zcount] == 0) zcount++;
    size_t size = (binsz - zcount) * 138 / 100 + 1;
    if (size > 220) return false;
    uint8_t buf[220];
    for (size_t k = 0; k < size; k++) buf[k] = 0;
    size_t i, j, high;
    for (i = zcount, high = size - 1; i < binsz; i++, high = j) {
        int carry = (int)bin[i];
        for (j = size - 1; (j > high) || carry; j--) {
            carry += 256 * buf[j];
            buf[j] = (uint8_t)(carry % 58);
            carry /= 58;
            if (!j) break;
        }
    }
    for (j = 0; j < size && !buf[j]; j++);
    size_t need = zcount + (size - j) + 1;
    if (b58cap < need) return false;
    i = 0;
    for (size_t k = 0; k < zcount; k++) b58[i++] = '1';
    for (; j < size; i++, j++) b58[i] = b58d[buf[j]];
    b58[i] = '\0';
    return true;
}

struct U256 {
    uint8_t b[32];
    void zero()  { memset(b, 0, 32); }
    void mul58() {
        uint32_t carry = 0;
        for (int i = 31; i >= 0; i--) {
            uint32_t v = (uint32_t)b[i] * 58 + carry;
            b[i] = (uint8_t)(v & 0xFF); carry = v >> 8;
        }
    }
    void addByte(uint8_t x) {
        uint32_t carry = x;
        for (int i = 31; i >= 0 && carry; i--) {
            uint32_t v = b[i] + carry; b[i] = (uint8_t)(v & 0xFF); carry = v >> 8;
        }
    }
    void add(const U256 &o) {
        uint32_t carry = 0;
        for (int i = 31; i >= 0; i--) {
            uint32_t v = b[i] + o.b[i] + carry; b[i] = (uint8_t)(v & 0xFF); carry = v >> 8;
        }
    }
};

static const char B58_ALPHA[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static int b58idx(char c) {
    for (int i = 0; i < 58; i++) if (B58_ALPHA[i] == c) return i;
    return -1;
}

static bool compute_range(const char *p, int k,
                          uint8_t lo_out[32], uint8_t hi_out[32]) {
    U256 lo, step; lo.zero();
    for (int i = 0; i < k; i++) {
        int idx = b58idx(p[i]); if (idx < 0) return false;
        lo.mul58(); lo.addByte((uint8_t)idx);
    }
    step.zero(); step.b[31] = 1;
    for (int i = 0; i < 44 - k; i++) step.mul58();
    for (int i = 0; i < 44 - k; i++) lo.mul58();
    U256 hi = lo; hi.add(step);
    memcpy(lo_out, lo.b, 32); memcpy(hi_out, hi.b, 32);
    return true;
}

static bool read_patterns_file(const char *path, std::vector<std::string> &out) {
    out.clear();
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line) && (int)out.size() < MAX_PATTERNS) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        out.push_back(line);
    }
    return !out.empty();
}

static std::string patterns_signature(const std::vector<std::string> &patterns) {
    std::string s;
    for (const auto &p : patterns) { s += p; s.push_back('\n'); }
    return s;
}

static bool upload_patterns(const std::vector<std::string> &patterns, bool quiet,
                            bool no_range_prefilter, std::string *err_out) {
    int count = (int)patterns.size();
    if (count < 1 || count > MAX_PATTERNS) {
        if (err_out) *err_out = "pattern count out of range";
        return false;
    }
    char    h_patterns[MAX_PATTERNS][MAX_PATTERN_LEN + 1] = {};
    int     h_lens[MAX_PATTERNS] = {};
    uint8_t h_lo[MAX_PATTERNS][32] = {};
    uint8_t h_hi[MAX_PATTERNS][32] = {};
    int     h_rv[MAX_PATTERNS] = {};

    for (int i = 0; i < count; i++) {
        const std::string &p = patterns[i];
        int len = (int)p.size();
        if (len > MAX_PATTERN_LEN) {
            if (err_out) *err_out = std::string("pattern too long: ") + p;
            return false;
        }
        strncpy(h_patterns[i], p.c_str(), MAX_PATTERN_LEN);
        h_patterns[i][MAX_PATTERN_LEN] = '\0';
        h_lens[i] = len;
        int solid = 0;
        while (solid < len && p[solid] != '?') solid++;
        if (no_range_prefilter)
            h_rv[i] = 0;
        else
            h_rv[i] = (solid > 0) && compute_range(p.c_str(), solid, h_lo[i], h_hi[i]) ? 1 : 0;
        if (!quiet)
            printf("Pattern[%d]: %s (len=%d, range_check=%s)\n", i, p.c_str(), len, h_rv[i] ? "yes" : "no");
    }

    cudaError_t e;
    e = cudaMemcpyToSymbol(g_patterns, h_patterns, sizeof(h_patterns));
    if (e != cudaSuccess) { if (err_out) *err_out = cudaGetErrorString(e); return false; }
    e = cudaMemcpyToSymbol(g_pattern_lens, h_lens, sizeof(h_lens));
    if (e != cudaSuccess) { if (err_out) *err_out = cudaGetErrorString(e); return false; }
    e = cudaMemcpyToSymbol(g_pattern_count, &count, sizeof(int));
    if (e != cudaSuccess) { if (err_out) *err_out = cudaGetErrorString(e); return false; }
    e = cudaMemcpyToSymbol(g_range_lo, h_lo, sizeof(h_lo));
    if (e != cudaSuccess) { if (err_out) *err_out = cudaGetErrorString(e); return false; }
    e = cudaMemcpyToSymbol(g_range_hi, h_hi, sizeof(h_hi));
    if (e != cudaSuccess) { if (err_out) *err_out = cudaGetErrorString(e); return false; }
    e = cudaMemcpyToSymbol(g_range_valid, h_rv, sizeof(h_rv));
    if (e != cudaSuccess) { if (err_out) *err_out = cudaGetErrorString(e); return false; }
    return true;
}

static void reset_match_flags_all_devices(int n_gpu) {
    int z = 0;
    for (int g = 0; g < n_gpu; g++) {
        cudaSetDevice(g);
        cudaMemcpyToSymbol(g_match_flag, &z, sizeof(int));
    }
}

static bool poll_match_and_print(int n_gpu, const VanityOpts &opts,
                                 const std::vector<std::string> &patterns_live,
                                 const std::vector<std::string> &request_ids) {
    for (int g = 0; g < n_gpu; g++) {
        cudaSetDevice(g);
        int fl = 0;
        cudaMemcpyFromSymbol(&fl, g_match_flag, sizeof(int));
        if (fl != 1) continue;

        int pi = 0;
        unsigned char seed[32], pub[32];
        cudaMemcpyFromSymbol(&pi, g_match_pi, sizeof(int));
        cudaMemcpyFromSymbol(seed, g_match_seed, 32);
        cudaMemcpyFromSymbol(pub, g_match_pub, 32);

        char addr[52], sec[128];
        if (!host_b58_encode(pub, 32, addr, sizeof(addr))) {
            fprintf(stderr, "host_b58_encode address failed\n");
            return true;
        }
        unsigned char sec64[64];
        memcpy(sec64, seed, 32);
        memcpy(sec64 + 32, pub, 32);
        if (!host_b58_encode(sec64, 64, sec, sizeof(sec))) {
            fprintf(stderr, "host_b58_encode secret failed\n");
            return true;
        }

        if (opts.worker_multi) {
            if (pi >= 0 && pi < (int)patterns_live.size())
                printf("matched_pattern=%s\n", patterns_live[pi].c_str());
            else
                printf("matched_pattern=\n");
            if (pi >= 0 && pi < (int)request_ids.size())
                printf("matched_request_id=%s\n", request_ids[pi].c_str());
            else
                printf("matched_request_id=\n");
        }
        printf("address=%s\n", addr);
        printf("private_key_bs58=%s\n", sec);
        fflush(stdout);
        return true;
    }
    return false;
}

static bool parse_bool(const char *s) {
    if (!s) return false;
    if (strcasecmp(s, "true") == 0 || strcmp(s, "1") == 0) return true;
    if (strcasecmp(s, "yes") == 0) return true;
    return false;
}

static void parse_patterns_arg(const char *s, std::vector<std::string> &patterns,
                               std::vector<std::string> &request_ids) {
    patterns.clear();
    request_ids.clear();
    std::string acc;
    for (const char *p = s; ; p++) {
        if (*p == ',' || *p == '\0') {
            if (!acc.empty()) {
                size_t eq = acc.find('=');
                if (eq != std::string::npos) {
                    request_ids.push_back(acc.substr(0, eq));
                    patterns.push_back(acc.substr(eq + 1));
                } else {
                    request_ids.push_back("");
                    patterns.push_back(acc);
                }
            }
            acc.clear();
            if (*p == '\0') break;
        } else {
            acc.push_back(*p);
        }
    }
}

int main(int argc, char const *argv[]) {
    std::vector<std::string> patterns;
    std::vector<std::string> request_ids;
    bool split_key = false;
    unsigned char user_point_bytes[32] = {};
    bool watch_mode = false;
    const char *watch_path = "/tmp/patterns.txt";
    VanityOpts opts;
    int case_sensitive_host = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--watch") == 0) {
            watch_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') watch_path = argv[++i];
        } else if (strcmp(argv[i], "--user-point") == 0 && i + 1 < argc) {
            const char *hex = argv[++i];
            if (strlen(hex) != 64) { fprintf(stderr, "--user-point must be 64 hex\n"); return 1; }
            for (int j = 0; j < 32; j++) {
                unsigned int bv = 0; sscanf(hex + j * 2, "%02x", &bv);
                user_point_bytes[j] = (unsigned char)bv;
            }
            split_key = true;
            if (!opts.quiet) printf("SplitKey: user_point=%s\n", hex);
        } else if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            patterns.push_back(argv[++i]);
            opts.suffix = false;
            opts.worker_output = true;
            opts.phantom = true;
        } else if (strcmp(argv[i], "--suffix") == 0 && i + 1 < argc) {
            patterns.push_back(argv[++i]);
            opts.suffix = true;
            opts.worker_output = true;
            opts.phantom = true;
        } else if (strcmp(argv[i], "--patterns") == 0 && i + 1 < argc) {
            parse_patterns_arg(argv[++i], patterns, request_ids);
            opts.worker_output = true;
            opts.worker_multi = true;
            opts.phantom = true;
        } else if (strcmp(argv[i], "--is-prefix") == 0 && i + 1 < argc) {
            opts.suffix = !parse_bool(argv[++i]);
        } else if (strcmp(argv[i], "--case-sensitive") == 0 && i + 1 < argc) {
            case_sensitive_host = parse_bool(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--stop-after-keys") == 0 && i + 1 < argc) {
            opts.stop_after_keys = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-iterations") == 0 && i + 1 < argc) {
            opts.max_iterations = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--attempts-per-execution") == 0 && i + 1 < argc) {
            opts.attempts_per_execution = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--quiet") == 0) {
            opts.quiet = true;
        } else if (strcmp(argv[i], "--phantom") == 0) {
            opts.phantom = true;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        } else if ((int)patterns.size() < MAX_PATTERNS) {
            patterns.push_back(std::string(argv[i]));
        }
    }

    if (opts.worker_multi && (int)request_ids.size() != (int)patterns.size()) {
        fprintf(stderr, "--patterns: internal parse error\n");
        return 1;
    }

    if (split_key && opts.worker_output) {
        fprintf(stderr, "Split-key mode is incompatible with worker-style output.\n");
        return 1;
    }

    if (patterns.empty()) {
        if (!read_patterns_file(watch_path, patterns) && !watch_mode) {
            fprintf(stderr,
                    "Usage: %s [options] pattern1 ...\n"
                    "  Worker: --prefix|--suffix PAT --case-sensitive t|f --stop-after-keys N ...\n"
                    "  Multi:  --patterns id=pat,... --is-prefix t|f ...\n",
                    argv[0]);
            return 1;
        }
    }
    if (patterns.empty()) {
        fprintf(stderr, "[watch] Waiting for non-empty %s (or pass patterns on CLI)\n", watch_path);
        while (!read_patterns_file(watch_path, patterns)) sleep(1);
    }

    for (const auto &p : patterns) {
        if (p == "--prefix" || p == "--suffix" || p == "--case-sensitive" || p == "--quiet" ||
            p == "--stop-after-keys" || p == "--max-iterations" || p == "--attempts-per-execution" ||
            p == "--patterns" || p == "--is-prefix" || p == "--phantom" || p == "--watch" ||
            p == "--user-point") {
            fprintf(stderr,
                    "Argument `%s` was read as a vanity pattern, not an option. "
                    "You are likely running an old `cuda_ed25519_vanity` (e.g. stale copy in bin/). "
                    "Rebuild: (cd cuda-miner/src && make clean && make V=debug debug/cuda_ed25519_vanity), "
                    "then copy debug/cuda_ed25519_vanity and debug/libcuda-crypt.so to bin/, "
                    "or run: LD_LIBRARY_PATH=cuda-miner/src/debug ./cuda-miner/src/debug/cuda_ed25519_vanity ...\n",
                    p.c_str());
            return 1;
        }
    }

    /* range_match uses the literal pattern; case-insensitive allows symbols outside that range. */
    opts.no_range_prefilter = opts.suffix || (case_sensitive_host == 0);
    std::string err;
    if (!upload_patterns(patterns, opts.quiet, opts.no_range_prefilter, &err)) {
        fprintf(stderr, "Invalid patterns: %s\n", err.c_str());
        return 1;
    }

    time_t watch_mtime0 = 0;
    if (watch_mode) {
        struct stat st;
        if (stat(watch_path, &st) == 0) watch_mtime0 = st.st_mtime;
    }
    const std::string patterns_sig0 = patterns_signature(patterns);

    int mode = split_key ? 1 : 0;
    cudaMemcpyToSymbol(g_split_key_mode, &mode, sizeof(int));
    if (split_key) {
        cudaMemcpyToSymbol(g_user_point, user_point_bytes, 32);
        if (!opts.quiet) printf("SplitKey mode enabled.\n");
    }

    int pm = (opts.phantom && !split_key) ? 1 : 0;
    cudaMemcpyToSymbol(g_phantom_mode, &pm, sizeof(int));
    int sm = opts.suffix ? 1 : 0;
    cudaMemcpyToSymbol(g_suffix_mode, &sm, sizeof(int));
    int wo = opts.worker_output ? 1 : 0;
    cudaMemcpyToSymbol(g_worker_output, &wo, sizeof(int));
    if (!opts.worker_output)
        case_sensitive_host = 1;
    cudaMemcpyToSymbol(g_case_sensitive, &case_sensitive_host, sizeof(int));

    {
        unsigned char s8[32] = {}; s8[0] = 8;
        ge_p3 sp; ge_cached sc;
        ge_scalarmult_base(&sp, s8); ge_p3_to_cached(&sc, &sp);
        cudaMemcpyToSymbol(g_step_cached, &sc, sizeof(ge_cached));
        if (!opts.quiet) printf("Step: 8·G precomputed.\n");
    }

    ed25519_set_verbose(!opts.quiet);

    int min_grid = 0, bs = 0, ma = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &bs, vanity_scan, 0, 0);
    (void)min_grid;
    cudaDeviceProp dp; cudaGetDeviceProperties(&dp, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ma, vanity_scan, bs, 0);
    int tb = ma * dp.multiProcessorCount;
    if (!opts.quiet)
        printf("Grid: %d blocks × %d threads = %d total threads  (BATCH_W=%d, ATTEMPTS=%d)\n",
               tb, bs, tb * bs, BATCH_W, ATTEMPTS);

    int default_batches = ATTEMPTS / BATCH_W;
    int batches_host = default_batches;
    if (opts.attempts_per_execution > 0) {
        int n_gpu_ct = 0;
        cudaGetDeviceCount(&n_gpu_ct);
        if (n_gpu_ct < 1) n_gpu_ct = 1;
        uint64_t total_th = (uint64_t)tb * (uint64_t)bs * (uint64_t)n_gpu_ct;
        uint64_t per_th = (opts.attempts_per_execution + total_th - 1) / total_th;
        if (per_th < (uint64_t)BATCH_W) per_th = BATCH_W;
        per_th = (per_th / BATCH_W) * BATCH_W;
        if (per_th / BATCH_W > (uint64_t)INT_MAX) batches_host = INT_MAX;
        else batches_host = (int)(per_th / BATCH_W);
    }
    cudaMemcpyToSymbol(g_batches, &batches_host, sizeof(int));

    if (watch_mode && !opts.quiet)
        printf("[watch] Hot-reload enabled: %s (updates apply after each kernel sync)\n", watch_path);

    gpu_config vanity;
    vanity_setup(vanity, tb, bs);
    vanity_run(vanity, tb, bs, watch_mode, watch_path, watch_mtime0, patterns_sig0, opts, patterns, request_ids);
    return 0;
}

void vanity_setup(gpu_config &vanity, int tb, int bs) {
    int n = 0; cudaGetDeviceCount(&n);
    for (int i = 0; i < n; i++) {
        cudaSetDevice(i); cudaDeviceProp d; cudaGetDeviceProperties(&d, i);
        printf("GPU[%d]: %s  SMs=%d — init %d curand states\n", i, d.name, d.multiProcessorCount, tb * bs);
        cudaMalloc((void **)&vanity.states[i], (size_t)tb * bs * sizeof(curandState));
        vanity_init<<<tb, bs>>>(vanity.states[i]);
    }
    cudaDeviceSynchronize();
    printf("END: Initializing Memory\n");
}

void vanity_run(gpu_config &vanity, int tb, int bs,
                bool watch, const char *watch_path,
                time_t watch_mtime0, const std::string &patterns_sig0,
                const VanityOpts &opts,
                const std::vector<std::string> &patterns_arg,
                const std::vector<std::string> &request_ids) {
    std::vector<std::string> live_patterns = patterns_arg;
    int n = 0; cudaGetDeviceCount(&n);
    int batches = 0;
    cudaMemcpyFromSymbol(&batches, g_batches, sizeof(int));
    if (batches <= 0) batches = ATTEMPTS / BATCH_W;
    long long apl = (long long)tb * bs * batches * BATCH_W;

    time_t last_mtime = watch_mtime0;
    std::string last_sig = patterns_sig0;

    int matches_found = 0;

    if (!opts.quiet) { printf("Warmup...\n"); fflush(stdout); }
    reset_match_flags_all_devices(n);
    for (int g = 0; g < n; g++) { cudaSetDevice(g); vanity_scan<<<tb, bs>>>(vanity.states[g]); }
    cudaDeviceSynchronize();
    if (poll_match_and_print(n, opts, live_patterns, request_ids)) {
        matches_found++;
        if (opts.stop_after_keys > 0 && matches_found >= opts.stop_after_keys)
            std::exit(0);
    }
    if (!opts.quiet) { printf("Warmup done. Mining...\n"); fflush(stdout); }

    for (long long iter = 0; iter < opts.max_iterations; iter++) {
        if (watch) {
            struct stat st;
            if (stat(watch_path, &st) == 0 && st.st_mtime != last_mtime) {
                last_mtime = st.st_mtime;
                std::vector<std::string> np;
                if (read_patterns_file(watch_path, np) && !np.empty()) {
                    std::string sig = patterns_signature(np);
                    if (sig != last_sig) {
                        std::string err;
                        if (upload_patterns(np, opts.quiet, opts.no_range_prefilter, &err)) {
                            last_sig = sig;
                            live_patterns = np;
                            if (!opts.quiet) printf("[watch] Reloaded %zu pattern(s)\n", np.size());
                        } else if (!opts.quiet)
                            fprintf(stderr, "[watch] Ignored invalid file: %s\n", err.c_str());
                    }
                } else if (!opts.quiet)
                    fprintf(stderr, "[watch] Ignored empty or unreadable %s\n", watch_path);
            }
        }

        reset_match_flags_all_devices(n);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int g = 0; g < n; g++) { cudaSetDevice(g); vanity_scan<<<tb, bs>>>(vanity.states[g]); }
        cudaDeviceSynchronize();
        double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        if (!opts.quiet) {
            printf("Iter %lld: %.2fM keys/s\n", iter, apl * n / dt / 1e6);
            fflush(stdout);
        }

        /* Need live pattern list for matched_pattern= */
        if (poll_match_and_print(n, opts, live_patterns, request_ids)) {
            matches_found++;
            if (opts.stop_after_keys > 0 && matches_found >= opts.stop_after_keys)
                std::exit(0);
        }
    }

    if (opts.worker_output && opts.stop_after_keys > 0 && matches_found < opts.stop_after_keys)
        std::exit(1);
}

/* ── Kernels ─────────────────────────────────────────────────────────────── */
void __global__ vanity_init(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long clk = clock64();
    unsigned long long seed64 = clk ^ ((unsigned long long)id * 0x9E3779B97F4A7C15ULL);
    curand_init(seed64, id, 0, &state[id]);
}

__device__ __forceinline__
int range_match(const uint8_t *a, const uint8_t *lo, const uint8_t *hi) {
    int ge = 1;
    for (int i = 0; i < 32; i++) {
        if (a[i] > lo[i]) { ge = 1; break; }
        if (a[i] < lo[i]) { ge = 0; break; }
    }
    if (!ge) return 0;
    for (int i = 0; i < 32; i++) {
        if (a[i] < hi[i]) return 1;
        if (a[i] > hi[i]) return 0;
    }
    return 0;
}

__device__ __forceinline__ char d_tolower_c(char c) {
    if (c >= 'A' && c <= 'Z') return (char)(c - 'A' + 'a');
    return c;
}

__device__ __forceinline__ bool pat_eq(char a, char b) {
    if (g_case_sensitive) return a == b;
    return d_tolower_c(a) == d_tolower_c(b);
}

__device__ int d_strlen(const char *s) {
    int n = 0;
    while (s[n] && n < MAX_PATTERN_LEN + 8) n++;
    return n;
}

__noinline__ __device__
void compute_first_point(ge_p3 *out, const unsigned char *seed,
                         const ge_cached *U_ptr, bool do_split) {
    ge_p3 W;
    ge_scalarmult_base(&W, seed);
    if (do_split) {
        ge_p1p1 tmp;
        ge_add(&tmp, &W, U_ptr);
        ge_p1p1_to_p3(out, &tmp);
    } else {
        *out = W;
    }
}

__noinline__ __device__
void check_match(const fe y_aff, int x_neg,
                 const unsigned char *secret32, bool do_split, bool phantom_mode) {
    unsigned char pub[32];
    fe_tobytes(pub, y_aff);
    pub[31] ^= (unsigned char)(x_neg) << 7;

    for (int pi = 0; pi < g_pattern_count; pi++) {
        bool cand = g_range_valid[pi]
            ? range_match(pub, g_range_lo[pi], g_range_hi[pi]) != 0
            : true;
        if (!cand) continue;

        char key[52] = {};
        size_t ksz = sizeof(key);
        /* Wallet-style address: base58-encoded compressed public key bytes. */
        b58enc(key, &ksz, pub, 32);
        int addr_len = d_strlen(key);
        int plen = g_pattern_lens[pi];
        int ok = 1;
        if (g_suffix_mode) {
            if (addr_len < plen) ok = 0;
            else {
                for (int j = 0; j < plen && ok; j++) {
                    if (g_patterns[pi][j] == '?') continue;
                    char cj = key[addr_len - plen + j];
                    if (!pat_eq(cj, g_patterns[pi][j])) ok = 0;
                }
            }
        } else {
            for (int j = 0; j < plen; j++) {
                if (g_patterns[pi][j] == '?') continue;
                if (j >= addr_len) { ok = 0; break; }
                if (!pat_eq(key[j], g_patterns[pi][j])) { ok = 0; break; }
            }
        }
        if (!ok) continue;

        if (do_split) {
            printf("MATCH_SK %d: %s - ", pi, key);
            for (int j = 0; j < 32; j++) printf("%02x", secret32[j]);
            printf("\n");
            continue;
        }

        if (g_worker_output) {
            /* 0=free, 2=claim (writing), 1=done. Never set 1 before payload + fence. */
            int st = atomicCAS(&g_match_flag, 0, 2);
            if (st == 0) {
                g_match_pi = pi;
                for (int j = 0; j < 32; j++) {
                    g_match_seed[j] = secret32[j];
                    g_match_pub[j] = pub[j];
                }
                __threadfence();
                atomicExch(&g_match_flag, 1);
            }
            return;
        }

        if (phantom_mode) {
            printf("PHANTOM %d: %s - ", pi, key);
            for (int j = 0; j < 32; j++) printf("%02x", secret32[j]);
            printf("\n");
        } else {
            printf("MATCH %d: %s - ", pi, key);
            for (int j = 0; j < 32; j++) printf("%02x", secret32[j]);
            printf("\n");
        }
        return;
    }
}

void __global__ vanity_scan(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState ls = state[id];

    int BATCHES = g_batches;
    if (BATCHES <= 0) BATCHES = ATTEMPTS / BATCH_W;

    bool do_split = (g_split_key_mode == 1);
    bool phantom = (g_phantom_mode == 1) && !do_split;

    ge_cached U_cached;
    if (do_split) {
        ge_p3 U_neg, U;
        ge_frombytes_negate_vartime(&U_neg, g_user_point);
        fe_neg(U.X, U_neg.X); fe_copy(U.Y, U_neg.Y);
        fe_copy(U.Z, U_neg.Z); fe_neg(U.T, U_neg.T);
        ge_p3_to_cached(&U_cached, &U);
    }

    fe bY[BATCH_W], bZ[BATCH_W], bX[BATCH_W];
    unsigned char bSeed[BATCH_W][32];
    fe zp[BATCH_W];

    if (phantom) {
        for (int bat = 0; bat < BATCHES; bat++) {
            for (int k = 0; k < BATCH_W; k++) {
                unsigned char seed_raw[32];
                for (int i = 0; i < 8; i++) {
                    uint32_t r = curand(&ls);
                    seed_raw[4 * i + 0] = (r >> 0) & 0xFF;
                    seed_raw[4 * i + 1] = (r >> 8) & 0xFF;
                    seed_raw[4 * i + 2] = (r >> 16) & 0xFF;
                    seed_raw[4 * i + 3] = (r >> 24) & 0xFF;
                }
                unsigned char hout[64];
                sha512(seed_raw, 32, hout);
                unsigned char sk[32];
                for (int j = 0; j < 32; j++) sk[j] = hout[j];
                sk[0] &= 248;
                sk[31] &= 63;
                sk[31] |= 64;

                ge_p3 P;
                ge_scalarmult_base(&P, sk);
                fe_copy(bY[k], P.Y);
                fe_copy(bZ[k], P.Z);
                fe_copy(bX[k], P.X);
                for (int j = 0; j < 32; j++) bSeed[k][j] = seed_raw[j];

                if (k == 0) fe_copy(zp[0], P.Z);
                else fe_mul(zp[k], zp[k - 1], P.Z);
            }

            fe inv_prod;
            fe_invert(inv_prod, zp[BATCH_W - 1]);
            fe t;
            fe_copy(t, inv_prod);

            for (int k = BATCH_W - 1; k >= 0; k--) {
                fe z_inv;
                if (k > 0) {
                    fe_mul(z_inv, t, zp[k - 1]);
                    fe_mul(t, t, bZ[k]);
                } else {
                    fe_copy(z_inv, t);
                }
                fe y_aff, x_aff;
                fe_mul(y_aff, bY[k], z_inv);
                fe_mul(x_aff, bX[k], z_inv);
                check_match(y_aff, fe_isnegative(x_aff), bSeed[k], false, true);
            }
        }
        state[id] = ls;
        return;
    }

    /* Legacy incremental path */
    unsigned char seed[32];
    for (int i = 0; i < 8; i++) {
        uint32_t r = curand(&ls);
        seed[4 * i + 0] = (r >> 0) & 0xFF; seed[4 * i + 1] = (r >> 8) & 0xFF;
        seed[4 * i + 2] = (r >> 16) & 0xFF; seed[4 * i + 3] = (r >> 24) & 0xFF;
    }
    seed[0] &= 248; seed[31] &= 63; seed[31] |= 64;

    ge_p3 check;
    compute_first_point(&check, seed, do_split ? &U_cached : nullptr, do_split);

    for (int bat = 0; bat < BATCHES; bat++) {
        for (int k = 0; k < BATCH_W; k++) {
            fe_copy(bY[k], check.Y);
            fe_copy(bZ[k], check.Z);
            fe_copy(bX[k], check.X);
            memcpy(bSeed[k], seed, 32);
            if (k == 0) fe_copy(zp[0], check.Z);
            else fe_mul(zp[k], zp[k - 1], check.Z);

            ge_p1p1 tmp;
            ge_add(&tmp, &check, &g_step_cached);
            ge_p1p1_to_p3(&check, &tmp);

            uint32_t carry = 8;
            for (int i = 0; i < 32 && carry; i++) {
                uint32_t v = seed[i] + carry; seed[i] = (unsigned char)(v & 0xFF); carry = v >> 8;
            }
            seed[31] = (seed[31] & 63) | 64;
        }

        fe inv_prod;
        fe_invert(inv_prod, zp[BATCH_W - 1]);
        fe t;
        fe_copy(t, inv_prod);

        for (int k = BATCH_W - 1; k >= 0; k--) {
            fe z_inv;
            if (k > 0) {
                fe_mul(z_inv, t, zp[k - 1]);
                fe_mul(t, t, bZ[k]);
            } else {
                fe_copy(z_inv, t);
            }
            fe y_aff, x_aff;
            fe_mul(y_aff, bY[k], z_inv);
            fe_mul(x_aff, bX[k], z_inv);
            check_match(y_aff, fe_isnegative(x_aff), bSeed[k], do_split, false);
        }
    }

    state[id] = ls;
}

bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz) {
    const char b58d[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    const uint8_t *bin = data; int carry; size_t i, j, high, zcount = 0, size;
    while (zcount < binsz && !bin[zcount]) zcount++;
    size = (binsz - zcount) * 138 / 100 + 1;
    uint8_t buf[64]; for (size_t k = 0; k < size; k++) buf[k] = 0;
    for (i = zcount, high = size - 1; i < binsz; i++, high = j)
        for (carry = bin[i], j = size - 1; (j > high) || carry; j--) {
            carry += 256 * buf[j]; buf[j] = carry % 58; carry /= 58; if (!j) break;
        }
    for (j = 0; j < size && !buf[j]; j++);
    if (*b58sz <= zcount + size - j) { *b58sz = zcount + size - j + 1; return false; }
    if (zcount) for (size_t k = 0; k < zcount; k++) b58[k] = '1';
    for (i = zcount; j < size; i++, j++) b58[i] = b58d[buf[j]];
    b58[i] = '\0'; *b58sz = i + 1; return true;
}
