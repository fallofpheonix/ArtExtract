#include "artextract/extractor.h"

#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace artextract {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

bool IsPrintableOrWhitespace(unsigned char byte) {
    if (byte == '\n' || byte == '\r' || byte == '\t' || byte == ' ') {
        return true;
    }
    return std::isprint(byte) != 0;
}

std::string EscapeJson(const std::string& input) {
    std::ostringstream out;
    for (unsigned char c : input) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << c;
                }
                break;
        }
    }
    return out.str();
}

}  // namespace

ExtractionResult ExtractFile(const std::filesystem::path& path) {
    if (path.empty()) {
        throw std::runtime_error("input path is empty");
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("input path does not exist: " + path.string());
    }
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("input path is not a regular file: " + path.string());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }

    ExtractionResult result;
    result.input_path = std::filesystem::absolute(path).string();

    std::array<std::uint64_t, 256> histogram{};
    std::uint64_t hash = kFnvOffset;
    std::vector<char> buffer(64 * 1024);

    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize read_count = input.gcount();
        if (read_count <= 0) {
            break;
        }

        result.bytes_total += static_cast<std::uint64_t>(read_count);
        for (std::streamsize i = 0; i < read_count; ++i) {
            const unsigned char byte = static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
            histogram[byte] += 1;
            if (byte == '\n') {
                result.lines_total += 1;
            }
            if (IsPrintableOrWhitespace(byte)) {
                result.printable_bytes += 1;
            }
            hash ^= static_cast<std::uint64_t>(byte);
            hash *= kFnvPrime;
        }
    }

    for (std::size_t i = 0; i < histogram.size(); ++i) {
        if (histogram[i] > result.dominant_byte_count) {
            result.dominant_byte_count = histogram[i];
            result.dominant_byte = static_cast<std::uint8_t>(i);
        }
    }

    if (result.bytes_total > 0) {
        result.printable_ratio = static_cast<double>(result.printable_bytes) /
                                 static_cast<double>(result.bytes_total);
    }

    std::ostringstream hash_hex;
    hash_hex << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << hash;
    result.hash_fnv1a64_hex = hash_hex.str();

    return result;
}

std::string ToJson(const ExtractionResult& result) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"input_path\": \"" << EscapeJson(result.input_path) << "\",\n";
    out << "  \"bytes_total\": " << result.bytes_total << ",\n";
    out << "  \"lines_total\": " << result.lines_total << ",\n";
    out << "  \"printable_bytes\": " << result.printable_bytes << ",\n";
    out << "  \"printable_ratio\": " << std::fixed << std::setprecision(6) << result.printable_ratio
        << ",\n";
    out << "  \"dominant_byte\": " << static_cast<unsigned int>(result.dominant_byte) << ",\n";
    out << "  \"dominant_byte_count\": " << result.dominant_byte_count << ",\n";
    out << "  \"hash_fnv1a64_hex\": \"" << result.hash_fnv1a64_hex << "\"\n";
    out << "}\n";
    return out.str();
}

}  // namespace artextract
