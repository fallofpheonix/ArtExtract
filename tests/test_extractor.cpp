#include "artextract/extractor.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int main() {
    int failures = 0;

    const fs::path tmp = fs::temp_directory_path() / "artextract_test_input.txt";
    {
        std::ofstream out(tmp, std::ios::binary);
        out << "ab\ncd\n";
    }

    try {
        const artextract::ExtractionResult result = artextract::ExtractFile(tmp);
        if (result.bytes_total != 6) {
            std::cerr << "bytes_total mismatch: expected 6, got " << result.bytes_total << "\n";
            failures += 1;
        }
        if (result.lines_total != 2) {
            std::cerr << "lines_total mismatch: expected 2, got " << result.lines_total << "\n";
            failures += 1;
        }
        if (result.printable_bytes != 6) {
            std::cerr << "printable_bytes mismatch: expected 6, got " << result.printable_bytes
                      << "\n";
            failures += 1;
        }
        if (result.dominant_byte != static_cast<std::uint8_t>('\n')) {
            std::cerr << "dominant_byte mismatch: expected newline (10), got "
                      << static_cast<unsigned int>(result.dominant_byte) << "\n";
            failures += 1;
        }
        if (result.hash_fnv1a64_hex.size() != 16) {
            std::cerr << "hash length mismatch: expected 16, got "
                      << result.hash_fnv1a64_hex.size() << "\n";
            failures += 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "unexpected extraction failure: " << e.what() << "\n";
        failures += 1;
    }

    try {
        (void)artextract::ExtractFile(tmp.string() + ".missing");
        std::cerr << "expected missing-file failure but extraction succeeded\n";
        failures += 1;
    } catch (const std::exception&) {
    }

    std::error_code ec;
    fs::remove(tmp, ec);

    if (failures > 0) {
        std::cerr << "test failures: " << failures << "\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
