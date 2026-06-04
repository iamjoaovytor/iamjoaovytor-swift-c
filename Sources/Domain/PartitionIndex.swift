import CSearch
import Foundation

public struct PartitionHeader {
    public static let magic: [UInt8] = [0x52, 0x50, 0x4B, 0x59] // "RPKY"
    public static let version: UInt32 = 1
    public static let bytes = 64
    public static let pageAlignment = 4096

    public let count: Int
    public let stride: Int
    public let dim: Int
}

public final class PartitionIndex: @unchecked Sendable {
    public let header: PartitionHeader
    public let offsets: UnsafeBufferPointer<UInt32>   // 257 entries (last = count)
    public let vectors: UnsafeBufferPointer<Int16>
    public let labels: UnsafeBufferPointer<UInt8>

    private let mappedBase: UnsafeMutableRawPointer
    private let mappedSize: Int

    init(
        header: PartitionHeader,
        offsets: UnsafeBufferPointer<UInt32>,
        vectors: UnsafeBufferPointer<Int16>,
        labels: UnsafeBufferPointer<UInt8>,
        mappedBase: UnsafeMutableRawPointer,
        mappedSize: Int
    ) {
        self.header = header
        self.offsets = offsets
        self.vectors = vectors
        self.labels = labels
        self.mappedBase = mappedBase
        self.mappedSize = mappedSize
    }

    deinit {
        munmap(mappedBase, mappedSize)
    }

    public enum LoadError: Error, CustomStringConvertible {
        case tooSmall
        case badMagic
        case badVersion(UInt32)
        case mmapFailed(errno: Int32)
        public var description: String {
            switch self {
            case .tooSmall: return "partition index file too small"
            case .badMagic: return "partition index bad magic"
            case .badVersion(let v): return "partition index bad version \(v)"
            case .mmapFailed(let e): return "partition index mmap failed errno=\(e)"
            }
        }
    }

    public static func defaultPath(for referencesPath: String) -> String {
        let url = URL(fileURLWithPath: referencesPath)
        return url.deletingPathExtension().appendingPathExtension("pkey").path
    }

    public static func load(path: String) throws -> PartitionIndex {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw LoadError.mmapFailed(errno: errno) }
        defer { close(fd) }

        var st = stat()
        fstat(fd, &st)
        let fileSize = Int(st.st_size)
        guard fileSize >= PartitionHeader.bytes else { throw LoadError.tooSmall }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: -1) else {
            throw LoadError.mmapFailed(errno: errno)
        }
        let rawBase = UnsafeRawPointer(mapped)

        let magic = rawBase.load(fromByteOffset: 0, as: (UInt8, UInt8, UInt8, UInt8).self)
        guard magic.0 == 0x52 && magic.1 == 0x50 && magic.2 == 0x4B && magic.3 == 0x59
        else { munmap(mapped, fileSize); throw LoadError.badMagic }

        let version = rawBase.load(fromByteOffset: 4, as: UInt32.self)
        guard version == PartitionHeader.version
        else { munmap(mapped, fileSize); throw LoadError.badVersion(version) }

        let count  = Int(rawBase.load(fromByteOffset: 8,  as: UInt64.self))
        let stride = Int(rawBase.load(fromByteOffset: 16, as: UInt32.self))
        let dim    = Int(rawBase.load(fromByteOffset: 20, as: UInt32.self))

        let header = PartitionHeader(count: count, stride: stride, dim: dim)

        var offset = PartitionHeader.bytes

        // 257 UInt32 offsets (256 partitions + sentinel)
        let offsetsPtr = rawBase.advanced(by: offset).bindMemory(to: UInt32.self, capacity: 257)
        let offsetsBuf = UnsafeBufferPointer(start: offsetsPtr, count: 257)
        offset += 257 * MemoryLayout<UInt32>.size
        offset = (offset + PartitionHeader.pageAlignment - 1) & ~(PartitionHeader.pageAlignment - 1)

        let vectorsPtr = rawBase.advanced(by: offset).bindMemory(to: Int16.self, capacity: count * stride)
        let vectorsBuf = UnsafeBufferPointer(start: vectorsPtr, count: count * stride)
        offset += count * stride * MemoryLayout<Int16>.size
        offset = (offset + PartitionHeader.pageAlignment - 1) & ~(PartitionHeader.pageAlignment - 1)

        let labelsPtr = rawBase.advanced(by: offset).bindMemory(to: UInt8.self, capacity: count)
        let labelsBuf = UnsafeBufferPointer(start: labelsPtr, count: count)

        return PartitionIndex(
            header: header,
            offsets: offsetsBuf,
            vectors: vectorsBuf,
            labels: labelsBuf,
            mappedBase: mapped,
            mappedSize: fileSize
        )
    }

    public func warm() {
        _ = madvise(mappedBase, mappedSize, MADV_WILLNEED)
        _ = madvise(mappedBase, mappedSize, MADV_HUGEPAGE)
        _ = mlock(UnsafeRawPointer(mappedBase), mappedSize)
        let rawBase = UnsafeRawPointer(mappedBase)
        var sink: UInt8 = 0
        var off = 0
        while off < mappedSize {
            sink &+= rawBase.load(fromByteOffset: off, as: UInt8.self)
            off += PartitionHeader.pageAlignment
        }
        _ = sink
    }

    /// Exact top-k search within a single partition using the filtered AVX2 kernel.
    public func fraudVoteCount(
        query: [Int16],
        partitionKey: UInt8,
        k: Int,
        worstDist: Int64 = Int64.max
    ) -> (fraudVotes: Int, worstDistOut: Int64) {
        let pk = Int(partitionKey)
        let start = Int(offsets[pk])
        let end   = Int(offsets[pk + 1])
        guard end > start else { return (0, worstDist) }

        let candidateCount = end - start
        let actualK = min(k, candidateCount)

        return withUnsafeTemporaryAllocation(of: rinha_neighbor_t.self, capacity: actualK) { rawNeighbors in
            query.withUnsafeBufferPointer { queryBuf in
                let vectorsBase = vectors.baseAddress!.advanced(by: start * header.stride)
                rinha_topk_exact_i16_filtered(
                    queryBuf.baseAddress,
                    vectorsBase,
                    numericCast(candidateCount),
                    numericCast(header.dim),
                    numericCast(header.stride),
                    numericCast(actualK),
                    rawNeighbors.baseAddress,
                    worstDist
                )
                var fraudVotes = 0
                var worst = worstDist
                for i in 0..<actualK {
                    let n = rawNeighbors[i]
                    guard n.record_index >= 0 else { continue }
                    let globalIdx = start + Int(n.record_index)
                    if labels[globalIdx] == 1 { fraudVotes += 1 }
                    if n.distance_squared < worst { worst = n.distance_squared }
                }
                return (fraudVotes, worst)
            }
        }
    }
}
