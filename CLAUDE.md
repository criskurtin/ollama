# CLAUDE.md - Ollama Codebase Guide for AI Assistants

> **Last Updated:** 2025-11-14
> **Repository:** https://github.com/ollama/ollama
> **Purpose:** Guide AI assistants in understanding and contributing to the Ollama codebase

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Codebase Structure](#codebase-structure)
3. [Development Setup](#development-setup)
4. [Architecture and Key Components](#architecture-and-key-components)
5. [Code Conventions and Patterns](#code-conventions-and-patterns)
6. [Testing Guidelines](#testing-guidelines)
7. [Contributing Standards](#contributing-standards)
8. [Common Development Tasks](#common-development-tasks)
9. [Important Notes for AI Assistants](#important-notes-for-ai-assistants)

---

## Repository Overview

**Ollama** is a sophisticated system for running large language models locally. It provides both a CLI and HTTP API server for interacting with LLMs, supporting multiple platforms (macOS, Windows, Linux) with GPU acceleration options.

### Key Technologies

- **Primary Language:** Go 1.24.1
- **Backend:** C/C++ (llama.cpp/GGML integration)
- **Web Framework:** Gin (HTTP server)
- **CLI Framework:** Cobra
- **Database:** SQLite3
- **Build System:** CMake + Go modules
- **Testing:** Go testing + integration tests

### Project Goals

- Enable local LLM inference with minimal friction
- Support multiple model architectures (Llama, Gemma, Mistral, QWen, Phi, etc.)
- Provide OpenAI-compatible API
- Maintain high performance with GPU acceleration
- Keep backward compatibility

---

## Codebase Structure

### Top-Level Directory Organization

```
ollama/
├── main.go                    # Entry point (13 lines - bootstraps CLI)
├── cmd/                       # CLI implementation (Cobra-based)
├── server/                    # HTTP API server (15k+ lines)
├── api/                       # Client library and API types
├── llm/                       # LLM server interface
├── ml/                        # Neural network backend (4.6k lines)
├── model/                     # Tokenization and encoding (15k+ lines)
├── convert/                   # Model format converters (23+ architectures)
├── runner/                    # Model execution runners
├── app/                       # Desktop application (18 subdirectories)
├── auth/                      # Authentication module
├── discover/                  # Hardware detection
├── fs/                        # File system utilities (GGUF/GGML parsing)
├── types/                     # Shared type definitions
├── integration/               # Integration tests
├── docs/                      # Documentation
├── CMakeLists.txt             # C/C++ build configuration (5.6k lines)
├── go.mod/go.sum              # Go dependencies
└── Dockerfile                 # Multi-stage container builds
```

### Core Packages by Function

#### Server & API (`/server`, `/api`)
- **server/routes.go** (65KB): Main HTTP server, Gin-based routing
- **server/sched.go** (29KB): Scheduler for model loading/unloading
- **server/manifest.go**: Model manifest management
- **server/layer.go**: Layer-based architecture for model storage
- **api/types.go** (34KB): Request/Response type definitions
- Endpoints: `/api/generate`, `/api/chat`, `/api/embed`, `/api/pull`, `/api/push`, `/api/create`, etc.

#### Command Line (`/cmd`)
- **cmd/cmd.go** (48KB): CLI commands implementation
- **cmd/interactive.go** (19KB): Interactive REPL
- Commands: `run`, `create`, `pull`, `push`, `delete`, `list`, `show`
- Platform-specific: `start_darwin.go`, `start_windows.go`, `start_linux.go`

#### Model Processing (`/model`, `/ml`, `/convert`)
- **model/**: Tokenization (BytePairEncoding, SentencePiece, WordPiece)
- **ml/nn/**: Neural network layers (embedding, attention, linear, pooling, RoPE)
- **ml/backend/ggml/**: GGML backend integration
- **convert/**: Model converters for 23+ architectures (Llama, Gemma, Mistral, etc.)

#### LLM Runtime (`/llm`, `/runner`)
- **llm/server.go** (54KB): LLM server interface
- Platform-specific implementations for GPU/CPU
- **runner/**: Execution runners (common, llama, ollama)

#### Utilities
- **discover/**: CPU/GPU detection (Linux, Windows)
- **fs/**: GGUF and GGML format parsing
- **sample/**: Sampling strategies (nucleus, temperature)
- **kvcache/**: Key-value cache for inference
- **template/**, **harmony/**: Prompt template handling
- **tools/**, **thinking/**: Advanced model features
- **progress/**: CLI progress bars and spinners
- **readline/**: Cross-platform terminal input

---

## Development Setup

### Prerequisites

1. **Go 1.24.1+** - https://go.dev/doc/install
2. **C/C++ Compiler:**
   - macOS: Clang (Xcode Command Line Tools)
   - Windows: TDM-GCC (amd64) or llvm-mingw (arm64)
   - Linux: GCC/Clang
3. **CMake** (except Windows ARM)
4. **Optional GPU Support:**
   - NVIDIA: CUDA SDK (11 or 12)
   - AMD: ROCm
   - Apple Silicon: Metal (built-in)

### Quick Start

```shell
# Clone and enter repository
cd ollama

# Run development server
go run . serve

# In another terminal, test
go run . run llama3.2

# Run tests
go test ./...

# Run integration tests
go test -tags=integration ./integration
```

### Platform-Specific Build

#### macOS (Apple Silicon)
```shell
go run . serve  # Metal support built-in
```

#### macOS (Intel) / Linux / Windows
```shell
cmake -B build
cmake --build build
go run . serve
```

#### Windows (ROCm)
```shell
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
go run . serve
```

### Library Detection

Ollama looks for acceleration libraries relative to the executable:
- `./lib/ollama` (Windows)
- `../lib/ollama` (Linux)
- `.` (macOS)
- `build/lib/ollama` (development)

### CGO Cache Issues

If experiencing crashes, force full rebuild:
```shell
go clean -cache
go run . serve
```

---

## Architecture and Key Components

### HTTP API Server (Gin-based)

**Location:** `server/routes.go`

The server is built on Gin web framework with:
- RESTful endpoints for model operations
- OpenAI-compatible middleware
- WebSocket support for streaming
- CORS middleware
- Scheduler integration for model lifecycle

**Key Endpoints:**
- `POST /api/generate` - Generate completions
- `POST /api/chat` - Chat completions
- `POST /api/embed` - Generate embeddings
- `POST /api/pull` - Download models
- `POST /api/push` - Upload models
- `POST /api/create` - Create custom models
- `DELETE /api/delete` - Delete models
- `GET /api/list` - List local models
- `POST /api/show` - Show model details

### Scheduler (`server/sched.go`)

Manages model loading/unloading with:
- VRAM-aware optimization (low VRAM threshold: 20GB)
- Context management for conversation memory
- Keep-alive timeouts
- GPU/CPU queue management
- Concurrent request handling

### Model Management

**Components:**
- **Layer-based storage:** Models stored as layers (blobs)
- **Manifest system:** Tracks model versions and configurations
- **Download/Upload:** Resumable downloads with progress tracking
- **Model path resolution:** Parse and resolve model names
- **Fixblobs operations:** Integrity checking and repair

### Advanced Features

1. **Multi-modal Support:** Image input processing for vision models
2. **Tool Use:** Integration of external tools/functions
3. **Extended Thinking:** Chain-of-thought reasoning support
4. **Sampling Strategies:** Nucleus sampling, temperature, top-k, top-p
5. **Template System:** Prompt template rendering with harmony format

---

## Code Conventions and Patterns

### 1. Commit Message Format

**Required Format:**
```
<package>: <short description>

- Package: Most affected Go package or directory
- Description: Lowercase start, continuation of "This changes Ollama to..."
```

**Good Examples:**
```
llm/backend/mlx: support the llama architecture
CONTRIBUTING: provide clarity on good commit messages
docs: simplify manual installation with shorter curl commands
```

**Bad Examples:**
```
feat: add more emoji
fix: was not using famous web framework
chore: generify code
```

### 2. Error Handling Patterns

**Custom Error Types:**
```go
// api/types.go
type StatusError struct {
    StatusCode int
    Status     string
    ErrorMessage string `json:"error"`
}

type AuthorizationError struct {
    SignInURL string
}
```

**Error Types Package:** `types/errtypes/`

**Convention:** Always check errors, but `errcheck` linter is disabled to allow unchecked errors where appropriate.

### 3. API Request/Response Pattern

**Struct Pairs:**
```go
type GenerateRequest struct {
    Model    string
    Prompt   string
    Stream   *bool  // Pointer for optional field
    Context  []int
    Options  map[string]interface{}
}

type GenerateResponse struct {
    Model     string
    Response  string
    Done      bool
    Context   []int
}
```

**Conventions:**
- Use pointers for optional fields (`*bool`, `*time.Duration`)
- Include `json:` tags for serialization
- `Stream` boolean for streaming support
- Separate Request/Response types

### 4. Platform-Specific Code

**File-based Approach:**
```
gpu_info_darwin.go     // macOS implementation
gpu_info_windows.go    // Windows implementation
gpu_info_linux.go      // Linux implementation
```

**Build Tags:**
```go
//go:build integration
```

### 5. Dependency Injection Pattern

**Constructor Functions:**
```go
func NewServer() *Server {
    return &Server{
        sched: NewScheduler(),
        // ...
    }
}
```

**Context-based Cancellation:**
- Use `context.Context` throughout
- Timeout pattern: `ctx, cancel := context.WithTimeout(context.Background(), duration)`

### 6. Type Safety

- Strong type system with Go structs
- Enum patterns via string constants with validation
- TypeScript generation from Go types
- Custom types for domain concepts (e.g., `ImageData []byte`)

### 7. Concurrency Patterns

- `golang.org/x/sync/errgroup` for parallel operations
- Atomic counters for concurrent access
- Scheduler manages GPU/CPU queue
- Context-aware operations

### 8. Progress Tracking

```go
import "github.com/ollama/ollama/progress"

bar := progress.NewBar(total)
bar.Set(current)

spinner := progress.NewSpinner()
spinner.Start()
defer spinner.Stop()
```

---

## Testing Guidelines

### Test Organization

**Three Types of Tests:**

1. **Unit Tests:** In-package `*_test.go` files
2. **Integration Tests:** `/integration` package with `//go:build integration` tag
3. **Benchmark Tests:** `*_benchmark_test.go` files

### Running Tests

```shell
# Unit tests
go test ./...

# Integration tests
go test -tags=integration ./integration

# Specific package
go test ./server

# With synctest (for Go 1.24 compatibility)
GOEXPERIMENT=synctest go test ./...

# Benchmarks
go test -bench=. ./sample
```

### Test Conventions

**Table-Driven Tests:**
```go
func TestSomething(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected string
    }{
        {"case1", "input1", "output1"},
        {"case2", "input2", "output2"},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Test logic
        })
    }
}
```

**Context with Timeout:**
```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()
```

**Resource-Aware Skipping:**
```go
func skipUnderMinVRAM(t *testing.T, minGB int) {
    // Skip if insufficient VRAM
}
```

**Streaming Tests:**
- Use `responseRecorder` pattern
- Test both stream=true and stream=false

### Integration Test Examples

- `basic_test.go`: Core functionality (Unicode, BlueSky)
- `api_test.go`: Full API testing
- `concurrency_test.go`: Multi-threaded requests
- `context_test.go`: Conversation context
- `tools_test.go`: Tool/function calling
- `model_perf_test.go`: Performance benchmarks

### Testing Tools

- Standard `testing` package
- `github.com/stretchr/testify` for assertions
- `github.com/google/go-cmp` for comparisons

---

## Contributing Standards

### Code Quality

**golangci-lint Configuration** (`.golangci.yaml`):
- Timeout: 5 minutes
- Enabled: asasalint, bidichk, bodyclose, gofmt, gofumpt, gosimple, govet, staticcheck, etc.
- Disabled: errcheck, usestdlibvars
- Severity: gofmt/goimports/intrange as info-level

**Run Linter:**
```shell
golangci-lint run
```

### Pull Request Guidelines

1. **Include Tests:** Strive to test behavior, not implementation
2. **Documentation:** Update docs if adding new features
3. **Backward Compatibility:** Don't break existing API
4. **Dependencies:** Add sparingly, justify necessity
5. **Discussion:** Open issue first for non-trivial changes

### What Gets Accepted

**Ideal Contributions:**
- Bug fixes (unexpected errors, crashes)
- Performance improvements
- Security fixes (report privately first - see SECURITY.md)

**Harder to Review:**
- New features (add surface area, maintenance burden)
- Large refactoring (takes longer to review)
- Extensive documentation additions

**May Not Be Accepted:**
- Breaking backward compatibility
- Adding significant user friction
- Creating large maintenance burden

### Proposing Non-Trivial Changes

Before opening PR, open issue with:
1. Problem you're solving (not what you're doing)
2. Why it's important
3. How it will be used
4. How it will be tested
5. Draft documentation

---

## Common Development Tasks

### Building from Source

```shell
# Quick development build
go run . serve

# Full build with CMake
cmake -B build
cmake --build build --parallel 8

# Docker build
docker build .

# Docker with ROCm
docker build --build-arg FLAVOR=rocm .
```

### Model Operations

```shell
# Pull a model
./ollama pull llama3.2

# Run interactive
./ollama run llama3.2

# Create custom model from Modelfile
./ollama create mymodel -f ./Modelfile

# List models
./ollama list

# Show model info
./ollama show llama3.2

# Remove model
./ollama rm llama3.2

# Copy model
./ollama cp llama3.2 my-model
```

### API Testing

```shell
# Start server
./ollama serve

# Generate (curl)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

### Environment Variables

Key environment variables:
- `OLLAMA_HOST` - Server binding (default: http://127.0.0.1:11434)
- `OLLAMA_EXPERIMENT` - Feature flags (comma-separated)
- `OLLAMA_*` - Various runtime configuration options
- `GOEXPERIMENT=synctest` - Enable Go 1.24 synctest package

### Debugging

```shell
# Enable verbose logging
OLLAMA_DEBUG=1 go run . serve

# Clear CGO cache on crashes
go clean -cache

# Check library detection
# Libraries should be in build/lib/ollama/ for development
```

---

## Important Notes for AI Assistants

### 1. Always Check Before Making Changes

- **Read existing code first** to understand patterns
- **Check tests** to understand expected behavior
- **Review CONTRIBUTING.md** for contribution guidelines
- **Consider backward compatibility** - API changes must not break existing users

### 2. File Location Conventions

When working on code:
- **Server logic:** `server/` package
- **API types:** `api/types.go`
- **CLI commands:** `cmd/cmd.go`
- **Model processing:** `model/` or `ml/` packages
- **Tests:** Co-located `*_test.go` files or `integration/` for integration tests

### 3. Common Pitfalls

**CGO Issues:**
- Data structures can change, causing crashes
- Solution: `go clean -cache`

**Platform-Specific Code:**
- Always consider Windows, macOS, Linux
- Use build tags or separate files (`*_darwin.go`, etc.)

**Testing:**
- Must pass `go test ./...`
- Must pass integration tests for significant changes
- CI uses `GOEXPERIMENT=synctest` - may need to test locally

**Dependencies:**
- Don't add unless absolutely necessary
- Explain why in PR

### 4. Code Navigation Tips

**Finding Functionality:**
- HTTP routes: `server/routes.go`
- Model loading: `server/sched.go`
- Tokenization: `model/` package
- CLI commands: `cmd/cmd.go`
- Type definitions: `api/types.go` and `types/` package

**Architecture References:**
- Model converters: `convert/` directory (23+ architectures)
- Neural net layers: `ml/nn/` directory
- Backend integration: `ml/backend/ggml/`

### 5. Security Considerations

- **Never commit secrets** (.env, credentials, etc.)
- Report security issues privately (see SECURITY.md)
- Validate user inputs
- Be aware of command injection, path traversal risks

### 6. Performance Considerations

- VRAM management is critical (low VRAM threshold: 20GB)
- Scheduler optimizes model loading/unloading
- Parallel operations use `errgroup`
- Progress tracking for long operations

### 7. API Compatibility

**Critical:** Maintain OpenAI-compatible API:
- `/v1/chat/completions`
- `/v1/embeddings`
- Match OpenAI request/response formats where applicable

### 8. Documentation Updates

When adding features:
- Update `docs/api.md` for API changes
- Update `README.md` if user-facing
- Consider updating `docs/development.md` for developer changes

### 9. Model Architecture Support

Current supported architectures (23+):
- Llama, Gemma, Mistral, QWen, Phi, GPT-NeoX, Falcon, StarCoder, etc.
- Each has converter in `convert/` directory
- Follow existing patterns when adding new architectures

### 10. Working with llama.cpp

- `llama/` directory contains submodule
- Sync with upstream via `Makefile.sync`
- Patches applied on top of upstream
- CMake integration in `CMakeLists.txt`

### 11. Helpful Commands Reference

```shell
# Quick test specific package
go test ./server -v

# Run with race detection
go test -race ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Format code
gofmt -w .

# Run linter
golangci-lint run

# Build for different platform
GOOS=linux GOARCH=amd64 go build

# See Go experiments
go env GOEXPERIMENT
```

### 12. When in Doubt

- Check Discord: https://discord.gg/ollama
- Review similar PRs and issues on GitHub
- Ask maintainers before major changes
- Prefer small, focused changes over large refactors

---

## Quick Reference: File Paths for Common Tasks

| Task | File Path | Lines |
|------|-----------|-------|
| Add HTTP endpoint | `server/routes.go` | 65KB |
| Add CLI command | `cmd/cmd.go` | 48KB |
| Add API type | `api/types.go` | 34KB |
| Model scheduling logic | `server/sched.go` | 29KB |
| LLM interface | `llm/server.go` | 54KB |
| Add tokenizer | `model/` package | 15k+ |
| Add model architecture | `convert/` directory | - |
| Add neural net layer | `ml/nn/` directory | - |
| Integration tests | `integration/` package | - |
| Development docs | `docs/development.md` | - |
| API docs | `docs/api.md` | - |

---

## Architecture Diagram (Simplified)

```
User
  ├─> CLI (cmd/) ────────────┐
  └─> HTTP API (server/) ────┤
                             │
                         Scheduler (sched.go)
                             │
                    ┌────────┴────────┐
                    │                 │
              Model Manager      LLM Runtime (llm/)
              (manifest, layers)      │
                    │            ┌────┴────┐
              File System       Runner    Backend
              (GGUF/GGML)    (runner/)   (GGML/llama.cpp)
                                          │
                                    ┌─────┴─────┐
                                  CPU/GPU    Device
                                (discover/)  Detection
```

---

## Version Information

- **Go Version:** 1.24.1
- **GOEXPERIMENT:** synctest (for CI compatibility)
- **Main Dependencies:**
  - Gin v1.10.0 (web framework)
  - Cobra v1.7.0 (CLI)
  - SQLite3 v1.14.24 (database)
  - go-cmp v0.7.0 (testing)
  - uuid v1.6.0 (ID generation)

---

## Additional Resources

- **Main README:** `/README.md`
- **Contributing Guide:** `/CONTRIBUTING.md`
- **Security Policy:** `/SECURITY.md`
- **Development Guide:** `/docs/development.md`
- **API Documentation:** `/docs/api.md`
- **Examples:** `/docs/examples.md`
- **Troubleshooting:** `/docs/troubleshooting.md`
- **Discord Community:** https://discord.gg/ollama
- **Reddit:** https://reddit.com/r/ollama

---

**Note:** This document is intended for AI assistants working on the Ollama codebase. It should be updated when significant architectural changes occur or new conventions are established.
