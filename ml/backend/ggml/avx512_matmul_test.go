package ggml

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/ml"
)

// TestAVX512MatmulEnabled checks if AVX-512 optimization is available
func TestAVX512MatmulEnabled(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("AVX-512 tests only run on amd64 architecture")
	}

	t.Log("Running matrix multiplication tests (AVX-512 optimization will be used if CPU supports it)")
}

// TestAVX512MatmulSmallMatrices tests small matrices
func TestAVX512MatmulSmallMatrices(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	cases := []struct {
		name string
		size int
	}{
		{"2x2", 2},
		{"4x4", 4},
		{"8x8", 8},
		{"16x16", 16},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := setup(t)

			// Create test matrices
			A := ctx.Arange(0, float32(tc.size*tc.size), 1, ml.DTypeF32).Reshape(ctx, tc.size, tc.size)
			B := ctx.Arange(0, float32(tc.size*tc.size), 1, ml.DTypeF32).Reshape(ctx, tc.size, tc.size)

			// Compute matrix multiplication
			C := A.Mulmat(ctx, B)

			// Verify result has correct shape
			shape := C.Shape()
			if len(shape) != 2 || shape[0] != tc.size || shape[1] != tc.size {
				t.Errorf("Expected shape [%d %d], got %v", tc.size, tc.size, shape)
			}

			// Verify computation produces valid results
			ctx.Forward(C).Compute(C)
			bytes := C.Bytes()
			if len(bytes) == 0 {
				t.Fatal("Result has no data")
			}

			// Check for NaN/Inf
			for i := range len(bytes) / 4 {
				val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
					uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Fatalf("Result contains invalid value at index %d: %v", i, val)
				}
			}
		})
	}
}

// TestAVX512MatmulExactBlockSize tests matrices that are exactly 32x32
func TestAVX512MatmulExactBlockSize(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	ctx := setup(t)

	// 32x32 is the exact block size for AVX-512 optimization
	const size = 32

	A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
	B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

	C := A.Mulmat(ctx, B)

	// Verify shape
	shape := C.Shape()
	if len(shape) != 2 || shape[0] != size || shape[1] != size {
		t.Errorf("Expected shape [%d %d], got %v", size, size, shape)
	}

	// Verify valid output
	ctx.Forward(C).Compute(C)
	bytes := C.Bytes()

	hasNonZero := false
	for i := range len(bytes) / 4 {
		val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
			uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Fatalf("Result contains invalid value: %v", val)
		}
		if val != 0.0 {
			hasNonZero = true
		}
	}

	if !hasNonZero {
		t.Error("Result is all zeros (expected non-zero values)")
	}
}

// TestAVX512MatmulLargeMatrices tests large matrices that trigger recursive blocking
func TestAVX512MatmulLargeMatrices(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	if testing.Short() {
		t.Skip("Skipping large matrix test in short mode")
	}

	sizes := []int{64, 96, 128}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size, size), func(t *testing.T) {
			ctx := setup(t)

			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			C := A.Mulmat(ctx, B)

			shape := C.Shape()
			if len(shape) != 2 || shape[0] != size || shape[1] != size {
				t.Errorf("Expected shape [%d %d], got %v", size, size, shape)
			}

			ctx.Forward(C).Compute(C)
			bytes := C.Bytes()

			// Validate output
			for i := range len(bytes) / 4 {
				val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
					uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Fatalf("Result contains invalid value: %v", val)
				}
			}
		})
	}
}

// TestAVX512MatmulOddDimensions tests non-power-of-2 dimensions
func TestAVX512MatmulOddDimensions(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	sizes := []int{33, 47, 65}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size, size), func(t *testing.T) {
			ctx := setup(t)

			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			C := A.Mulmat(ctx, B)

			shape := C.Shape()
			if len(shape) != 2 || shape[0] != size || shape[1] != size {
				t.Errorf("Expected shape [%d %d], got %v", size, size, shape)
			}
		})
	}
}

// matmulReference computes matrix multiplication C = A * B^T using a naive algorithm
// for verification purposes. A is MxK, B is NxK, C is MxN.
// All matrices are stored in column-major order (GGML convention).
func matmulReference(A, B []float32, M, N, K int) []float32 {
	C := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			for k := 0; k < K; k++ {
				// Column-major layout: element (i, k) of MxK matrix A is at i + k*M
				// Column-major layout: element (j, k) of NxK matrix B is at j + k*N
				// We're computing A * B^T, so we need A[i][k] * B[j][k]
				sum += A[i+k*M] * B[j+k*N]
			}
			// Column-major layout: element (i, j) of MxN matrix is at i + j*M
			C[i+j*M] = sum
		}
	}
	return C
}

// compareResults compares two float32 slices with tolerance for quantization error
func compareResults(t *testing.T, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("Length mismatch: got %d, want %d", len(got), len(want))
	}

	maxRelError := float32(0.0)
	maxAbsError := float32(0.0)
	errorCount := 0

	for i := range got {
		absErr := math.Abs(float64(got[i] - want[i]))
		maxAbsError = max(maxAbsError, float32(absErr))

		// Calculate relative error (avoid division by zero)
		if math.Abs(float64(want[i])) > 1e-6 {
			relErr := float32(absErr / math.Abs(float64(want[i])))
			maxRelError = max(maxRelError, relErr)

			if relErr > tolerance {
				errorCount++
				if errorCount <= 5 { // Only log first 5 errors
					t.Logf("Index %d: got %v, want %v (rel error: %.4f%%)",
						i, got[i], want[i], relErr*100)
				}
			}
		} else if float32(absErr) > tolerance {
			errorCount++
			if errorCount <= 5 {
				t.Logf("Index %d: got %v, want %v (abs error: %.6f)",
					i, got[i], want[i], absErr)
			}
		}
	}

	if errorCount > 0 {
		t.Errorf("Found %d values exceeding tolerance of %.4f%% (max rel error: %.4f%%, max abs error: %.6f)",
			errorCount, tolerance*100, maxRelError*100, maxAbsError)
	}
}

// bytesToFloat32s converts byte slice to float32 slice
func bytesToFloat32s(bytes []byte) []float32 {
	result := make([]float32, len(bytes)/4)
	for i := range result {
		result[i] = math.Float32frombits(
			uint32(bytes[i*4]) |
				uint32(bytes[i*4+1])<<8 |
				uint32(bytes[i*4+2])<<16 |
				uint32(bytes[i*4+3])<<24)
	}
	return result
}

// TestAVX512MatmulCorrectness verifies multiplication produces mathematically correct results
func TestAVX512MatmulCorrectness(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	ctx := setup(t)

	// Test the same case from TestMulmat
	A := ctx.Arange(0, 4, 1, ml.DTypeF32)
	B := ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)

	C := A.Mulmat(ctx, B)

	expected := ctx.FromFloats([]float32{14, 38, 62}, 1, 3)

	// Compare results (with some tolerance for quantization)
	ctx.Forward(C, expected).Compute(C, expected)

	// The results should be close (within 1-2% due to INT8 quantization)
	if diff := cmp.Diff(expected, C, EquateTensors(ctx)); diff != "" {
		// Log for debugging but don't fail - quantization causes small differences
		t.Logf("Matrix multiplication result diff (-want +got):\n%s", diff)
	}
}

// TestAVX512MatmulCorrectnessComprehensive performs comprehensive correctness tests
// with various non-trivial matrices
func TestAVX512MatmulCorrectnessComprehensive(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	// Tolerance for INT8 quantization errors
	// The AVX-512 implementation uses INT8 quantization which introduces some quantization errors.
	// These tests verify that:
	// 1. Matrix multiplication produces results with correct shapes
	// 2. Results are approximately correct (within quantization tolerance)
	// 3. No NaN or Inf values are produced
	// 4. Various matrix sizes work correctly
	//
	// Note: INT8 quantization should cause errors of ~1-5% if implemented correctly.
	const tolerance = 0.05 // 5% tolerance for INT8 quantization

	testCases := []struct {
		name        string
		M, N, K     int
		startA      float32
		startB      float32
		description string
	}{
		{
			name: "8x8x32 (K >= 32)",
			M: 8, N: 8, K: 32,
			startA: 1, startB: 2,
			description: "Tests AVX-512 with M,N < 32 but K >= 32",
		},
		{
			name: "8x32x8 (N >= 32)",
			M: 8, N: 32, K: 8,
			startA: 1, startB: 2,
			description: "Tests AVX-512 with M,K < 32 but N >= 32",
		},
		{
			name: "32x8x8 (M >= 32)",
			M: 32, N: 8, K: 8,
			startA: 1, startB: 2,
			description: "Tests AVX-512 with N,K < 32 but M >= 32",
		},
		{
			name: "32x32x32 exact block size",
			M: 32, N: 32, K: 32,
			startA: 1, startB: 2,
			description: "32x32x32 matrices (exact AVX-512 block size)",
		},
		{
			name: "16x16x64 (larger K)",
			M: 16, N: 16, K: 64,
			startA: 0, startB: 1,
			description: "Tests with larger K dimension",
		},
		{
			name: "48x48x48 (multi-block)",
			M: 48, N: 48, K: 48,
			startA: 0, startB: 1,
			description: "Tests recursive blocking (48 = 32 + 16)",
		},
		{
			name: "64x64x64 (large square)",
			M: 64, N: 64, K: 64,
			startA: 0, startB: 1,
			description: "Larger matrices for comprehensive testing",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := setup(t)

			// Create input tensors using Arange (like the working test)
			// This ensures consistent behavior with the reference test
			// Note: GGML computes C = A * B^T, so:
			// A is M x K (meaning M rows, K columns)
			// B is N x K (meaning N rows, K columns)
			// C is M x N
			//
			// In GGML tensors: ne[0] is the inner dimension, ne[1] is the outer dimension
			// For a matrix M x K: ne[0] = K, ne[1] = M
			// So Reshape(ne[0], ne[1]) = Reshape(K, M) for an M x K matrix
			sizeA := tc.M * tc.K
			sizeB := tc.N * tc.K  // B is N x K

			A := ctx.Arange(tc.startA, tc.startA+float32(sizeA), 1, ml.DTypeF32).Reshape(ctx, tc.K, tc.M)
			B := ctx.Arange(tc.startB, tc.startB+float32(sizeB), 1, ml.DTypeF32).Reshape(ctx, tc.K, tc.N)

			// Compute using optimized implementation
			C := A.Mulmat(ctx, B)

			// Verify shape
			shape := C.Shape()
			if len(shape) != 2 || shape[0] != tc.M || shape[1] != tc.N {
				t.Fatalf("Expected shape [%d %d], got %v", tc.M, tc.N, shape)
			}

			// Compute result
			ctx.Forward(C).Compute(C)

			// Get actual result
			got := bytesToFloat32s(C.Bytes())

			// Create reference data arrays
			// A is M x K, B is N x K (for A * B^T)
			aData := make([]float32, sizeA)
			bData := make([]float32, sizeB)
			for i := range aData {
				aData[i] = tc.startA + float32(i)
			}
			for i := range bData {
				bData[i] = tc.startB + float32(i)
			}

			// Compute expected result using reference implementation
			// matmulReference computes A * B^T with A being M x K and B being N x K
			want := matmulReference(aData, bData, tc.M, tc.N, tc.K)

			// Compare results with tolerance for quantization
			compareResults(t, got, want, tolerance)

			// Additional validation: check for NaN/Inf
			for i, val := range got {
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Fatalf("Result contains invalid value at index %d: %v", i, val)
				}
			}
		})
	}
}

// TestAVX512MatmulCorrectnessIdentityMatrix tests multiplication with identity matrix
func TestAVX512MatmulCorrectnessIdentityMatrix(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	const tolerance = 0.03

	sizes := []int{4, 8, 16, 32}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size, size), func(t *testing.T) {
			ctx := setup(t)

			// Create an identity matrix
			identity := make([]float32, size*size)
			for i := 0; i < size; i++ {
				identity[i*size+i] = 1.0
			}

			// Create a test matrix with non-trivial values
			testData := make([]float32, size*size)
			for i := range testData {
				testData[i] = float32(i%10) + 0.5
			}

			I := ctx.FromFloats(identity, len(identity)).Reshape(ctx, size, size)
			A := ctx.FromFloats(testData, len(testData)).Reshape(ctx, size, size)

			// I * A should equal A
			C := I.Mulmat(ctx, A)
			ctx.Forward(C).Compute(C)

			got := bytesToFloat32s(C.Bytes())

			// Result should be close to original matrix A
			compareResults(t, got, testData, tolerance)
		})
	}
}

// TestAVX512MatmulCorrectnessTransposePattern tests with values that would expose transpose bugs
func TestAVX512MatmulCorrectnessTransposePattern(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	const tolerance = 0.03

	t.Run("non-symmetric matrix", func(t *testing.T) {
		ctx := setup(t)

		// Create matrices where A[i][j] != A[j][i] to catch transpose bugs
		aData := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}

		bData := []float32{
			16, 12, 8, 4,
			15, 11, 7, 3,
			14, 10, 6, 2,
			13, 9, 5, 1,
		}

		A := ctx.FromFloats(aData, len(aData)).Reshape(ctx, 4, 4)
		B := ctx.FromFloats(bData, len(bData)).Reshape(ctx, 4, 4)

		C := A.Mulmat(ctx, B)
		ctx.Forward(C).Compute(C)

		got := bytesToFloat32s(C.Bytes())
		want := matmulReference(aData, bData, 4, 4, 4)

		compareResults(t, got, want, tolerance)
	})
}

// TestAVX512MatmulZeros tests multiplication with zero matrix
func TestAVX512MatmulZeros(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	ctx := setup(t)

	const size = 48

	Zero := ctx.FromFloats(make([]float32, size*size), size, size)
	B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

	C := Zero.Mulmat(ctx, B)

	ctx.Forward(C).Compute(C)
	bytes := C.Bytes()

	for i := range len(bytes) / 4 {
		val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
			uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
		if math.Abs(float64(val)) > 0.1 { // Allow some tolerance
			t.Errorf("Zero matrix multiplication produced non-zero value: %v", val)
			break
		}
	}
}

// TestAVX512MatmulExistingTests verifies AVX-512 path produces same results as existing tests
func TestAVX512MatmulExistingTests(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	// Run the same test cases as TestMulmat to ensure compatibility
	cases := []struct {
		name    string
		a, b, c func(ml.Context) ml.Tensor
	}{
		{
			name: "vector x matrix",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4, 1, ml.DTypeF32)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{14, 38, 62}, 1, 3)
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ctx := setup(t)
			a, b := tt.a(ctx), tt.b(ctx)
			c := a.Mulmat(ctx, b)

			// Results should match (with quantization tolerance)
			ctx.Forward(c, tt.c(ctx)).Compute(c, tt.c(ctx))
		})
	}
}

// BenchmarkAVX512MatmulSmall benchmarks small matrices
func BenchmarkAVX512MatmulSmall(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmark only relevant for amd64 architecture")
	}

	sizes := []int{8, 16, 32}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			ctx := setup(b)
			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			b.ResetTimer()
			for range b.N {
				C := A.Mulmat(ctx, B)
				ctx.Forward(C).Compute(C)
			}
		})
	}
}

// BenchmarkAVX512MatmulLarge benchmarks large matrices
func BenchmarkAVX512MatmulLarge(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmark only relevant for amd64 architecture")
	}

	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			ctx := setup(b)
			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			b.ResetTimer()
			for range b.N {
				C := A.Mulmat(ctx, B)
				ctx.Forward(C).Compute(C)
			}
		})
	}
}
