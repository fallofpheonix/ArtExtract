#include "artextract/extractor.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void PrintUsage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " extract --input <path> [--output <path>]\n"
        << "  " << program << " --help\n";
}

int RunExtract(int argc, char** argv) {
    std::filesystem::path input_path;
    std::filesystem::path output_path;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --input");
            }
            input_path = argv[++i];
            continue;
        }
        if (arg == "--output") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --output");
            }
            output_path = argv[++i];
            continue;
        }
        throw std::runtime_error("unknown argument: " + arg);
    }

    if (input_path.empty()) {
        throw std::runtime_error("--input is required");
    }

    const artextract::ExtractionResult result = artextract::ExtractFile(input_path);
    const std::string json = artextract::ToJson(result);
    std::cout << json;

    if (!output_path.empty()) {
        std::ofstream out(output_path, std::ios::binary);
        if (!out.good()) {
            throw std::runtime_error("failed to open output file: " + output_path.string());
        }
        out << json;
        if (!out.good()) {
            throw std::runtime_error("failed to write output file: " + output_path.string());
        }
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 2;
    }

    const std::string command = argv[1];
    if (command == "--help" || command == "-h" || command == "help") {
        PrintUsage(argv[0]);
        return 0;
    }

    try {
        if (command == "extract") {
            return RunExtract(argc, argv);
        }
        throw std::runtime_error("unknown command: " + command);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
