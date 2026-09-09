// Single-process, Linux main-thread probe for llama.cpp b10516. Build against
// verified b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9 headers AND libraries.
// The caller verifies artifacts/backends, binds the governor to the host, and
// wraps every native stage (including init, enumeration and cleanup) in a lease
// plus NativeDeadline. No warmup, tokenization, normalization, fit or admission.
// All int APIs return 0 on success, -1 on error; consult cfetch_probe_error().
// device_count returns a nonnegative count. Strings returned remain borrowed.
#include "llama.h"
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr const char * revision = "b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9";
thread_local char error_text[1024] = {};
pid_t owner = 0;
bool initialized = false, model_ready = false, context_ready = false;
llama_model * model = nullptr;
llama_context * context = nullptr;
ggml_backend_dev_t devices[2] = {};
ggml_backend_reg_t loaded[2] = {};
uint32_t requested = 0;
struct close_file { void operator()(FILE * file) const noexcept { std::fclose(file); } };
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
void main_thread() {
    require(syscall(SYS_gettid) == getpid() && (!owner || owner == getpid()),
            "requires the original process main thread");
}
void ready() { main_thread(); require(initialized, "backend is not initialized"); }
template<class F> int protect(F body) noexcept {
    error_text[0] = '\0';
    try { return body(); }
    catch (const std::exception & e) { std::snprintf(error_text, sizeof(error_text), "%s", e.what()); }
    catch (...) { std::snprintf(error_text, sizeof(error_text), "unknown native exception"); }
    return -1;
}
template<size_t N> void copy(char (&out)[N], const char * value) {
    value = value ? value : "";
    require(std::strlen(value) < N, "device property exceeds ABI capacity");
    std::memcpy(out, value, std::strlen(value) + 1);
}
void metadata(const char * key, const char * expected) {
    char value[128] = {};
    require(llama_model_meta_val_str(model, key, value, sizeof(value)) >= 0 &&
            std::strcmp(value, expected) == 0, "unexpected model metadata");
}
}

extern "C" {
// ctypes.Structure field order/types are exactly as below (native alignment).
struct cfetch_probe_device {
    char name[128], backend[128], description[256], device_id[128];
    uint64_t memory_free, memory_total;
    int32_t type;
};
const char * cfetch_probe_error() noexcept { return error_text; }
const char * cfetch_probe_revision() noexcept { return revision; }
const char * cfetch_probe_runtime_version() noexcept { return llama_version(); }

int cfetch_probe_init(const char * cpu_library, const char * vulkan_library) noexcept {
    return protect([&] {
        main_thread(); require(!owner, "initialization is once per process"); owner = getpid();
        const char * paths[2] = {cpu_library, vulkan_library};
        const char * names[2] = {"CPU", "Vulkan"};
        for (int i = 0; i < 2; ++i) if (paths[i]) {
            require(paths[i][0] == '/', "backend library must be an explicit absolute path");
            require(!ggml_backend_reg_by_name(names[i]), "backend already registered");
            loaded[i] = ggml_backend_load(paths[i]);
            require(loaded[i] && std::strcmp(ggml_backend_reg_name(loaded[i]), names[i]) == 0,
                    "backend library does not match its requested kind");
        }
        require(ggml_backend_reg_by_name("CPU"), "explicit CPU backend required");
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i));
            require(std::strcmp(name, "CPU") == 0 || std::strcmp(name, "Vulkan") == 0,
                    "unexpected backend registered");
        }
        llama_backend_init(); // Nonempty registry prevents implicit load_all/search.
        initialized = true; return 0;
    });
}
int cfetch_probe_device_count() noexcept {
    return protect([] { ready(); const auto n = ggml_backend_dev_count();
        require(n <= INT_MAX, "too many devices"); return static_cast<int>(n); });
}
int cfetch_probe_device_info(int32_t index, cfetch_probe_device * output) noexcept {
    return protect([&] {
        ready(); require(output && index >= 0 && size_t(index) < ggml_backend_dev_count(), "invalid device index/output");
        auto dev = ggml_backend_dev_get(index); ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props); cfetch_probe_device result{};
        copy(result.name, props.name); copy(result.description, props.description);
        copy(result.backend, ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
        copy(result.device_id, props.device_id); result.type = props.type;
        result.memory_free = props.memory_free; result.memory_total = props.memory_total;
        *output = result; return 0;
    });
}
// Borrow a read-only, already-hashed regular-file fd; dup shares its seek offset.
// Caller keeps file immutable and rechecks its digest. No pathname reopen/mmap.
int cfetch_probe_load(int verified_fd, const char * exact_device) noexcept {
    return protect([&] {
        ready(); require(!model && exact_device && *exact_device, "model already loaded or missing device");
        struct stat st{}; const int flags = fcntl(verified_fd, F_GETFL);
        require(flags >= 0 && (flags & O_ACCMODE) == O_RDONLY &&
                fstat(verified_fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0,
                "model fd must be a nonempty read-only regular file");
        const bool cpu = std::strcmp(exact_device, "CPU") == 0;
        devices[0] = nullptr;
        if (!cpu) for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (std::strcmp(ggml_backend_dev_name(dev), exact_device) != 0) continue;
            require(!devices[0] && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                    std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Vulkan") == 0,
                    "device must uniquely name a discrete Vulkan GPU"); devices[0] = dev;
        }
        require(cpu || devices[0], "named Vulkan device is unavailable");
        const int fd = fcntl(verified_fd, F_DUPFD_CLOEXEC, 0); require(fd >= 0, "model fd duplication failed");
        FILE * file = fdopen(fd, "rb"); if (!file) close(fd);
        require(file, "model fdopen failed"); std::unique_ptr<FILE, close_file> stream(file);
        require(fseeko(file, 0, SEEK_SET) == 0, "model rewind failed");
        auto p = llama_model_default_params(); p.devices = devices;
        p.n_gpu_layers = cpu ? 0 : INT_MAX; p.main_gpu = 0; p.split_mode = LLAMA_SPLIT_MODE_NONE;
        // Artifact verification is external; check_tensors launches unbounded async validators.
        p.load_mode = LLAMA_LOAD_MODE_NONE; p.check_tensors = false; p.use_extra_bufts = false;
        model = llama_model_load_from_file_ptr(file, p); require(model, "model load failed; inspect native stderr");
        metadata("general.architecture", "gemma-embedding");
        metadata("gemma-embedding.dense_2_feat_in", "768"); metadata("gemma-embedding.dense_2_feat_out", "3072");
        metadata("gemma-embedding.dense_3_feat_in", "3072"); metadata("gemma-embedding.dense_3_feat_out", "768");
        require(llama_model_n_embd(model) == 768 && llama_model_n_embd_out(model) == 768 &&
                llama_model_has_decoder(model) && !llama_model_has_encoder(model), "unexpected model dimensions/type");
        model_ready = true; return 0;
    });
}
int cfetch_probe_context(uint32_t bucket) noexcept {
    return protect([&] {
        ready(); require(model_ready && !context, "model not ready or context already allocated");
        require(bucket == 32 || bucket == 64 || bucket == 128 || bucket == 257 ||
                bucket == 512 || bucket == 1024 || bucket == 2048, "invalid canonical bucket");
        auto p = llama_context_default_params(); p.n_ctx = bucket; p.n_batch = p.n_ubatch = bucket;
        p.n_seq_max = 1; p.n_threads = p.n_threads_batch = 4; p.embeddings = true;
        p.pooling_type = LLAMA_POOLING_TYPE_MEAN; p.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        p.type_k = p.type_v = GGML_TYPE_F32; p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        p.offload_kqv = p.op_offload = devices[0] != nullptr; p.no_perf = true;
        requested = bucket; context = llama_init_from_model(model, p);
        require(context, "context creation failed; inspect native stderr");
        require(llama_n_ctx(context) == ((bucket + 255) / 256) * 256 &&
                llama_n_ctx_seq(context) == llama_n_ctx(context) && llama_n_seq_max(context) == 1 &&
                llama_n_batch(context) == bucket && llama_n_ubatch(context) == bucket &&
                llama_pooling_type(context) == LLAMA_POOLING_TYPE_MEAN &&
                llama_n_threads(context) == 4 && llama_n_threads_batch(context) == 4, "unexpected actual context parameters");
        context_ready = true; return 0;
    });
}
// Eight uint32 values: requested, ctx, ctx_seq, batch, ubatch, seq_max, threads, threads_batch.
int cfetch_probe_context_info(uint32_t * output, size_t count) noexcept {
    return protect([&] { ready(); require(context && output && count == 8, "invalid context info request");
        const uint32_t values[] = {requested, llama_n_ctx(context), llama_n_ctx_seq(context), llama_n_batch(context),
            llama_n_ubatch(context), llama_n_seq_max(context), uint32_t(llama_n_threads(context)), uint32_t(llama_n_threads_batch(context))};
        std::memcpy(output, values, sizeof(values)); return 0; });
}
int cfetch_probe_infer(const int32_t * input, int32_t count, float * output, size_t dimensions) noexcept {
    return protect([&] {
        ready(); require(context_ready && input && output && dimensions == 768 && count > 0 &&
                uint32_t(count) <= requested, "invalid inference input/context/output");
        const auto vocab = llama_model_get_vocab(model); const int32_t vocabulary = llama_vocab_n_tokens(vocab);
        std::array<llama_token, 2048> tokens{}; std::array<llama_pos, 2048> positions{};
        std::array<int32_t, 2048> sequences{}; std::array<llama_seq_id *, 2048> ids{};
        std::array<int8_t, 2048> logits{}; llama_seq_id zero = 0;
        for (int32_t i = 0; i < count; ++i) {
            require(input[i] >= 0 && input[i] < vocabulary, "token outside model vocabulary");
            tokens[i] = input[i]; positions[i] = i; sequences[i] = 1; ids[i] = &zero; logits[i] = 1;
        }
        context_ready = false; // Any native failure requires explicit context disposal.
        if (auto memory = llama_get_memory(context)) llama_memory_clear(memory, false);
        llama_batch batch{count, tokens.data(), nullptr, positions.data(), sequences.data(), ids.data(), logits.data()};
        const int result = llama_decode(context, batch); // Exactly one decode; never retry/split/warm up.
        require(result == 0, "llama_decode failed; inspect native stderr");
        const float * values = llama_get_embeddings_seq(context, 0); // Synchronizes native work.
        require(values, "missing pooled embedding");
        for (size_t i = 0; i < dimensions; ++i) require(std::isfinite(values[i]), "nonfinite embedding");
        std::memcpy(output, values, dimensions * sizeof(float)); context_ready = true; return 0;
    });
}
int cfetch_probe_free_context() noexcept {
    return protect([] { ready(); if (context) llama_free(context);
        context = nullptr; context_ready = false; requested = 0; return 0; });
}
int cfetch_probe_free_model() noexcept {
    return protect([] { ready(); require(!context, "free context before model"); if (model) llama_model_free(model);
        model = nullptr; model_ready = false; devices[0] = nullptr; return 0; });
}
int cfetch_probe_shutdown() noexcept {
    return protect([] { main_thread(); require(owner && !context && !model, "free model/context before shutdown");
        if (initialized) llama_backend_free();
        initialized = false;
        for (int i = 1; i >= 0; --i) if (loaded[i]) { ggml_backend_unload(loaded[i]); loaded[i] = nullptr; }
        return 0; });
}
}
