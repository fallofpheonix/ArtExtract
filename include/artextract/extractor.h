#ifndef ARTEXTRACT_EXTRACTOR_H
#define ARTEXTRACT_EXTRACTOR_H

#include <cstdint>
#include <filesystem>
#include <string>

namespace artextract {

struct ExtractionResult {
    std::string input_path;
    std::uint64_t bytes_total = 0;
    std::uint64_t lines_total = 0;
    std::uint64_t printable_bytes = 0;
    double printable_ratio = 0.0;
    std::uint8_t dominant_byte = 0;
    std::uint64_t dominant_byte_count = 0;
    std::string hash_fnv1a64_hex;
};

ExtractionResult ExtractFile(const std::filesystem::path& path);
std::string ToJson(const ExtractionResult& result);

}  // namespace artextract

#endif
