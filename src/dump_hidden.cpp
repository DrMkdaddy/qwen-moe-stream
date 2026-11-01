// Dump the post-attention residual stream (ffn_inp) of every layer.
// This is a tiny llama.cpp consumer used only for calibrating the MoE router.
//
// build:
//   g++ -O2 -std=c++17 dump_hidden.cpp \
//     -I$LLAMA/include -I$LLAMA/ggml/include \
//     -L$LLAMA/build/bin -lllama -lggml -lggml-base -lggml-cpu \
//     -Wl,-rpath,$LLAMA/build/bin -o dump_hidden
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct state {
    std::string out_dir;
    std::vector<bool> saved;
    int n_layer = 0;
    int n_tokens = 0;
};

static int layer_of(const char * name) {
    if (std::strncmp(name, "ffn_inp-", 8) != 0) return -1;
    return std::atoi(name + 8);
}

static bool callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<state *>(user_data);
    int layer = layer_of(t->name);
    if (layer < 0) return false;
    if (ask) {
        return layer < (int) st->saved.size() && !st->saved[layer];
    }
    if (layer >= (int) st->saved.size() || st->saved[layer]) return true;
    if (t->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "unexpected dtype %s for %s\n", ggml_type_name(t->type), t->name);
        return true;
    }
    const int64_t d = t->ne[0];
    const int64_t nt = t->ne[1];
    std::vector<float> buf((size_t) d * nt);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/layer_%02d.f32", st->out_dir.c_str(), layer);
    FILE * f = std::fopen(path, "wb");
    if (!f) { std::perror(path); return true; }
    std::fwrite(buf.data(), sizeof(float), buf.size(), f);
    std::fclose(f);
    st->saved[layer] = true;
    st->n_tokens = (int) nt;
    std::fprintf(stderr, "saved %s d=%lld tokens=%lld\n", path, (long long) d, (long long) nt);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s model.gguf prompt.txt out_dir n_ctx\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt_path = argv[2];
    const char * out_dir = argv[3];
    int n_ctx = std::atoi(argv[4]);
    fs::create_directories(out_dir);

    std::ifstream pf(prompt_path);
    std::string prompt((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_layer = llama_model_n_layer(model);

    std::vector<llama_token> tokens(prompt.size() + 8);
    bool add_bos = llama_vocab_get_add_bos(vocab);
    int n = llama_tokenize(vocab, prompt.data(), (int) prompt.size(), tokens.data(), (int) tokens.size(), add_bos, true);
    if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 3; }
    tokens.resize(n);
    if (n > n_ctx) { tokens.resize(n_ctx); n = n_ctx; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    static state st;
    st.out_dir = out_dir;
    st.saved.assign(n_layer, false);
    st.n_layer = n_layer;
    cp.cb_eval = callback;
    cp.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    std::vector<int8_t> want_logits(n, 1);  // keep every token; stops llama from slicing the last layer
    batch.logits = want_logits.data();
    int rc = llama_decode(ctx, batch);
    std::fprintf(stderr, "decode rc=%d tokens=%d layers_saved=%d\n", rc, n, (int) std::count(st.saved.begin(), st.saved.end(), true));

    std::ofstream meta(std::string(out_dir) + "/meta.json");
    meta << "{\"n_layer\":" << n_layer << ",\"n_tokens\":" << st.n_tokens << ",\"n_ctx\":" << n_ctx << "}\n";
    meta.close();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc == 0 ? 0 : 5;
}
