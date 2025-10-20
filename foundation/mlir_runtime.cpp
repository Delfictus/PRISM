// mlir_runtime.cpp - MLIR Runtime C API Implementation
//
// Provides C API for MLIR JIT compilation and execution on GPU

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>

#include <cuda_runtime.h>
#include <memory>
#include <string>

extern "C" {

// ============================================================================
// Context Management
// ============================================================================

struct MlirContextWrapper {
    mlir::MLIRContext* context;
    mlir::DialectRegistry registry;
};

void* mlir_context_create() {
    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto* wrapper = new MlirContextWrapper();
    wrapper->context = new mlir::MLIRContext();

    // Register all dialects
    mlir::registerAllDialects(wrapper->registry);
    wrapper->context->appendDialectRegistry(wrapper->registry);
    wrapper->context->loadAllAvailableDialects();

    return wrapper;
}

void mlir_context_destroy(void* ctx) {
    auto* wrapper = static_cast<MlirContextWrapper*>(ctx);
    delete wrapper->context;
    delete wrapper;
}

// ============================================================================
// Module Operations
// ============================================================================

void* mlir_module_create_from_string(void* ctx, const char* mlir_code) {
    auto* wrapper = static_cast<MlirContextWrapper*>(ctx);

    // Parse MLIR source
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        mlir_code,
        mlir::ParserConfig(wrapper->context)
    );

    if (!module) {
        return nullptr;
    }

    return module.release().getOperation();
}

void mlir_module_destroy(void* module) {
    auto* op = static_cast<mlir::Operation*>(module);
    op->destroy();
}

// ============================================================================
// JIT Compilation
// ============================================================================

struct JitWrapper {
    std::unique_ptr<mlir::ExecutionEngine> engine;
};

void* mlir_jit_create(void* ctx) {
    auto* wrapper = static_cast<MlirContextWrapper*>(ctx);

    // Setup JIT target machine for GPU
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        return nullptr;
    }

    auto tmBuilder = *tmBuilderOrError;
    tmBuilder.setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);

    auto* jit = new JitWrapper();
    return jit;
}

void mlir_jit_destroy(void* jit) {
    auto* wrapper = static_cast<JitWrapper*>(jit);
    delete wrapper;
}

int mlir_jit_compile_module(void* jit, void* module) {
    auto* jitWrapper = static_cast<JitWrapper*>(jit);
    auto* moduleOp = mlir::cast<mlir::ModuleOp>(static_cast<mlir::Operation*>(module));

    // Setup optimization pipeline
    mlir::PassManager pm(moduleOp->getContext());

    // Add GPU optimization passes
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::createGpuAsyncRegionPass());
    pm.addPass(mlir::createGpuToLLVMConversionPass());

    // Run optimizations
    if (mlir::failed(pm.run(moduleOp))) {
        return -1;
    }

    // Create execution engine
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/3,
        /*sizeLevel=*/0,
        /*targetMachine=*/nullptr
    );

    auto engineOrError = mlir::ExecutionEngine::create(moduleOp, options);
    if (!engineOrError) {
        return -1;
    }

    jitWrapper->engine = std::move(*engineOrError);
    return 0;
}

// ============================================================================
// Execution
// ============================================================================

int mlir_jit_execute(
    void* jit,
    const char* function_name,
    void** args,
    size_t num_args,
    void* result
) {
    auto* jitWrapper = static_cast<JitWrapper*>(jit);

    if (!jitWrapper->engine) {
        return -1;
    }

    // Lookup function
    auto* fn = jitWrapper->engine->lookup(function_name);
    if (!fn) {
        return -1;
    }

    // Create invocation
    auto invocationResult = jitWrapper->engine->invokePacked(
        function_name,
        llvm::MutableArrayRef<void*>(args, num_args)
    );

    if (invocationResult) {
        return -1;
    }

    return 0;
}

// ============================================================================
// GPU Memory Management
// ============================================================================

void* mlir_gpu_alloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void mlir_gpu_free(void* ptr) {
    cudaFree(ptr);
}

int mlir_gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int mlir_gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"